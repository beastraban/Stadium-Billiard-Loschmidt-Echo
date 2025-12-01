%% Stadium Billiard Loschmidt Echo - Publication Figure
% 6-panel figure

clear; clc; close all;

%% Load data
load('stadium_grid.mat')

% Convert to double
T_values = double(T_values(:));
M_values = double(M_values(:));
delta_values = double(delta_values(:));
fidelity_grid = double(fidelity_grid);
epsilon = double(epsilon);

%% Nature Physics color palette
% Muted but with color: cream -> teal -> deep blue
cmap_nature = [
    0.98 0.96 0.90;   % cream
    0.90 0.88 0.75;   % light tan
    0.70 0.82 0.78;   % sage
    0.45 0.72 0.75;   % teal
    0.30 0.55 0.70;   % steel blue
    0.20 0.40 0.58;   % medium blue
    0.15 0.25 0.45;   % dark blue
    0.10 0.15 0.30;   % navy
];
% Interpolate to 256 colors
x_cmap = linspace(0, 1, size(cmap_nature, 1));
xi_cmap = linspace(0, 1, 256);
cmap_nature = interp1(x_cmap, cmap_nature, xi_cmap);

%% Find indices for specific values
% M = 461
[~, M_idx] = min(abs(M_values - 461));
M_plot = M_values(M_idx);

% delta = 0.185
[~, delta_idx] = min(abs(delta_values - 0.185));
delta_plot = delta_values(delta_idx);

fprintf('Using M = %d (index %d)\n', M_plot, M_idx);
fprintf('Using delta = %.4f (index %d)\n', delta_plot, delta_idx);

%% Create figure
fig = figure('Position', [50 50 1400 900], 'Color', 'w');

%% Subplot (2,3,1) - Contour plot at M=461 (T vs delta)
subplot(2,3,1)

F_slice_M = squeeze(fidelity_grid(M_idx, :, :));  % (n_delta x n_T)
[T_mesh, delta_mesh] = meshgrid(T_values, delta_values);

contourf(T_mesh, log10(delta_mesh), F_slice_M, 20, 'LineWidth', 0.5);
hold on;
contour(T_mesh, log10(delta_mesh), F_slice_M, [0.5 0.5], 'k-', 'LineWidth', 2);

colormap(cmap_nature);
caxis([0 1]);
xlabel('Time T', 'FontSize', 12);
ylabel('log_{10}(\delta)', 'FontSize', 12);
title(sprintf('(a) Fidelity contours, M = %d', M_plot), 'FontSize', 12);

%% Subplot (2,3,2) - Contour plot at delta=0.185 (T vs M)
subplot(2,3,2)

F_slice_delta = squeeze(fidelity_grid(:, delta_idx, :));  % (n_M x n_T)
[T_mesh2, M_mesh] = meshgrid(T_values, M_values);

contourf(T_mesh2, log10(M_mesh), F_slice_delta, 20, 'LineWidth', 0.5);
hold on;
contour(T_mesh2, log10(M_mesh), F_slice_delta, [0.5 0.5], 'k-', 'LineWidth', 2);

colormap(cmap_nature);
caxis([0 1]);
xlabel('Time T', 'FontSize', 12);
ylabel('log_{10}(M)', 'FontSize', 12);
title(sprintf('(b) Fidelity contours, \\delta = %.3f', delta_plot), 'FontSize', 12);

%% Subplot (2,3,3) - Critical time fit with error bars
subplot(2,3,3)

% Compute t_c for each (M, delta) pair
t_c_all = zeros(length(M_values), length(delta_values));

for i = 1:length(M_values)
    for j = 1:length(delta_values)
        F = squeeze(fidelity_grid(i, j, :));
        idx = find(F < 0.5, 1, 'first');
        if ~isempty(idx) && idx > 1
            t_c_all(i, j) = interp1(F(idx-1:idx), T_values(idx-1:idx), 0.5);
        elseif isempty(idx)
            t_c_all(i, j) = NaN;  % Never crosses
        else
            t_c_all(i, j) = NaN;  % Immediate failure
        end
    end
end

% Average over M (since t_c should be M-independent)
t_c_mean = nanmean(t_c_all, 1);
t_c_std = nanstd(t_c_all, 0, 1);

% x-axis: ln(delta/epsilon)
ln_delta_eps = log(delta_values / epsilon);

% Plot all points with error bars
errorbar(ln_delta_eps, t_c_mean, t_c_std, 'ko', 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.2 0.2 0.2], 'LineWidth', 1.5, 'CapSize', 4);
hold on;

% Exclude saturated points from fit: t_c > 0.8 * T_max
T_max = max(T_values);
saturation_threshold = 0.8 * T_max;
valid = ~isnan(t_c_mean) & (t_c_mean < saturation_threshold);

% Mark excluded points
saturated = ~isnan(t_c_mean) & (t_c_mean >= saturation_threshold);
plot(ln_delta_eps(saturated), t_c_mean(saturated), 'o', 'MarkerSize', 8, ...
    'LineWidth', 2, 'Color', [0.6 0.6 0.6], 'MarkerFaceColor', [0.8 0.8 0.8]);

% Linear fit on valid points only - DARK GREEN
dark_green = [0.0 0.4 0.2];
p = polyfit(ln_delta_eps(valid), t_c_mean(valid)', 1);
x_fit = linspace(min(ln_delta_eps(valid)), max(ln_delta_eps(valid)), 100);
plot(x_fit, polyval(p, x_fit), '-', 'Color', dark_green, 'LineWidth', 2.5);

% Extrapolate to show where saturation deviates
x_extrap = linspace(min(ln_delta_eps), max(ln_delta_eps), 100);
plot(x_extrap, polyval(p, x_extrap), '--', 'Color', dark_green, 'LineWidth', 1.5);

% Add saturation line (black dashed)
yline(saturation_threshold, 'k--', 'LineWidth', 1.5);
text(1, saturation_threshold + 1, sprintf('Saturation'), ...
    'FontSize', 10, 'Color', [0.3 0.3 0.3]);

xlabel('ln(\delta/\epsilon)', 'FontSize', 12);
ylabel('Critical time t_c', 'FontSize', 12);
title(sprintf('(c) t_c vs resolution, 1/\\lambda = %.2f', p(1)), 'FontSize', 12);
legend('Data', 'Saturated', sprintf('Fit: slope = %.2f', p(1)), ...
    'Location', 'northwest');
grid on;
box on;

% Compute mean relative error
rel_err = t_c_std(valid) ./ t_c_mean(valid);
mean_rel_err = nanmean(rel_err) * 100;

% Add text showing error bars are tiny
text(0.95, 0.15, sprintf('Mean error: %.1f%%\n(M-independent)', mean_rel_err), ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', 'FontSize', 9, ...
    'BackgroundColor', 'w', 'EdgeColor', [0.5 0.5 0.5]);

%% Subplot (2,3,4) - 3D surface of panel (a): T vs delta
subplot(2,3,4)

surf(T_mesh, log10(delta_mesh), F_slice_M, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
hold on;
contour3(T_mesh, log10(delta_mesh), F_slice_M, [0.5 0.5], 'k-', 'LineWidth', 2);

colormap(cmap_nature);
caxis([0 1]);
xlabel('Time T', 'FontSize', 11);
ylabel('log_{10}(\delta)', 'FontSize', 11);
zlabel('Fidelity', 'FontSize', 11);
title(sprintf('(d) Surface, M = %d', M_plot), 'FontSize', 12);
view(45, 25);

%% Subplot (2,3,5) - 3D surface of panel (b): T vs M
subplot(2,3,5)

surf(T_mesh2, log10(M_mesh), F_slice_delta, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
hold on;
contour3(T_mesh2, log10(M_mesh), F_slice_delta, [0.5 0.5], 'k-', 'LineWidth', 2);

colormap(cmap_nature);
caxis([0 1]);
xlabel('Time T', 'FontSize', 11);
ylabel('log_{10}(M)', 'FontSize', 11);
zlabel('Fidelity', 'FontSize', 11);
title(sprintf('(e) Surface, \\delta = %.3f', delta_plot), 'FontSize', 12);
view(45, 25);

% Single colorbar for all panels
cb = colorbar('Position', [0.92 0.35 0.015 0.5]);
cb.Label.String = 'Fidelity';
cb.Label.FontSize = 12;

%% Subplot (2,3,6) - Fidelity decay curves for different deltas (showing sigmoid shape)
subplot(2,3,6)

% Pick a few delta values to show
delta_show_idx = round(linspace(5, length(delta_values)-3, 5));
colors_lines = [
    0.85 0.35 0.10;   % orange
    0.50 0.70 0.20;   % olive
    0.30 0.55 0.70;   % steel blue
    0.55 0.35 0.65;   % purple
    0.20 0.20 0.20;   % dark gray
];

for k = 1:length(delta_show_idx)
    j = delta_show_idx(k);
    % Average fidelity over all M
    F_avg = squeeze(mean(fidelity_grid(:, j, :), 1));
    plot(T_values, F_avg, '-', 'Color', colors_lines(k,:), 'LineWidth', 2);
    hold on;
end

% Add legend
legend_str = cell(length(delta_show_idx), 1);
for k = 1:length(delta_show_idx)
    legend_str{k} = sprintf('\\delta = %.2f', delta_values(delta_show_idx(k)));
end
legend(legend_str, 'Location', 'northeast', 'FontSize', 9);

xlabel('Time T', 'FontSize', 12);
ylabel('Fidelity', 'FontSize', 12);
title('(f) Fidelity decay (sigmoid shape)', 'FontSize', 12);
ylim([0 1.05]);
xlim([0 T_max]);
grid on;
box on;

% Add F=0.5 reference line
yline(0.5, 'k--', 'LineWidth', 1);

% Print fit results
fprintf('\nFit results (excluding saturated points):\n');
fprintf('  Slope = %.3f (= 1/lambda)\n', p(1));
fprintf('  Lambda = %.3f\n', 1/p(1));
fprintf('  Excluded %d saturated points (t_c > %.1f)\n', sum(saturated), saturation_threshold);
fprintf('  Mean relative error: %.2f%% (t_c is M-independent)\n', mean_rel_err);

%% Save
saveas(gcf, 'loschmidt_6panel.png');
saveas(gcf, 'loschmidt_6panel.fig');
fprintf('Saved: loschmidt_6panel.png/fig\n');
