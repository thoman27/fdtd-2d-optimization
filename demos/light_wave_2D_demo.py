from LightWave2D.grid import Grid
from LightWave2D.experiment import Experiment
from MPSPlots import colormaps
from TypedUnit import ureg

grid = Grid(
    resolution=0.1 * ureg.micrometer,
    size_x=32 * ureg.micrometer,
    size_y=20 * ureg.micrometer,
    n_steps=300
)

experiment = Experiment(grid=grid)
'''
scatterer = experiment.add_circle(
    position=('30%', '50%'),
    epsilon_r=2,
    radius=3 * ureg.micrometer
)

source = experiment.add_line_source(
    wavelength=1550 * ureg.nanometer,
    position_0=('10%', '100%'),
    position_1=('10%', '0%'),
    amplitude=10,
)

experiment.add_pml(order=1, width="10%", sigma_max=5000 * ureg.siemens / ureg.meter)
'''

source = experiment.add_point_source(
    wavelength=1550 * ureg.nanometer,
    position =('50%', '50%'),
    amplitude=1,
)

experiment.run()

animation = experiment.render_propagation(
    skip_frame=5,
    colormap=colormaps.polytechnique.red_black_blue
)

animation.save('./spherical_scatterer.gif', writer='Pillow', fps=10)