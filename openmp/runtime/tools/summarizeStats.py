#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import sys
import os
import argparse
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

"""
Read the stats file produced by the OpenMP runtime
and produce a processed summary

The radar_factory original code was taken from
matplotlib.org/examples/api/radar_chart.html
We added support to handle negative values for radar charts
"""

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with num_vars axes."""
    # calculate evenly-spaced axis angles
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    #theta += np.pi/2

    def draw_poly_frame(self, x0, y0, r):
        # TODO: use transforms to convert (x, y) to (r, theta)
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_frame(self, x0, y0, r):
        return plt.Circle((x0, y0), r)

    frame_dict = {'polygon': draw_poly_frame, 'circle': draw_circle_frame}
    if frame not in frame_dict:
        raise ValueError, 'unknown value for `frame`: %s' % frame

    class RadarAxes(PolarAxes):
        """
        Class for creating a radar chart (a.k.a. a spider or star chart)

        http://en.wikipedia.org/wiki/Radar_chart
        """
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_frame = frame_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            #for line in lines:
            #    self._close_line(line)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels,fontsize=14)

        def _gen_axes_patch(self):
            x0, y0 = (0.5, 0.5)
            r = 0.5
            return self.draw_frame(x0, y0, r)

    register_projection(RadarAxes)
    return theta

# Code to read the raw stats
def extractSI(s):
    """Convert a measurement with a range suffix into a suitably scaled value"""
    du     = s.split()
    num    = float(du[0])
    units  = du[1] if len(du) == 2 else ' '
    # http://physics.nist.gov/cuu/Units/prefixes.html
    factor = {'Y':  1e24,
              'Z':  1e21,
              'E':  1e18,
              'P':  1e15,
              'T':  1e12,
              'G':  1e9,
              'M':  1e6,
              'k':  1e3,
              ' ':  1  ,
              'm': -1e3, # Yes, I do mean that, see below for the explanation.
              'u': -1e6,
              'n': -1e9,
              'p': -1e12,
              'f': -1e15,
              'a': -1e18,
              'z': -1e21,
              'y': -1e24}[units[0]]
    # Minor trickery here is an attempt to preserve accuracy by using a single
    # divide, rather than  multiplying by 1/x, which introduces two roundings
    # since 1/10 is not representable perfectly in IEEE floating point. (Not
    # that this really matters, other than for cleanliness, since we're likely
    # reading numbers with at most five decimal digits of precision).
    return  num*factor if factor > 0 else num/-factor

def readData(f):
    line = f.readline()
    fieldnames = [x.strip() for x in line.split(',')]
    line = f.readline().strip()
    data = []
    while line != "":
        if line[0] != '#':
            fields = line.split(',')
            data.append ((fields[0].strip(), [extractSI(v) for v in fields[1:]]))
        line = f.readline().strip()
    # Man, working out this next incantation out was non-trivial!
    # They really want you to be snarfing data in csv or some other
    # format they understand!
    res = pd.DataFrame.from_items(data, columns=fieldnames[1:], orient='index')
    return res

def readTimers(f):
    """Skip lines with leading #"""
    line = f.readline()
    while line[0] == '#':
        line = f.readline()
    line = line.strip()
    if line == "Statistics on exit\n" or "Aggregate for all threads\n":
        line = f.readline()
    return readData(f)

def readCounters(f):
    """This can be just the same!"""
    return readData(f)

def readFile(fname):
    """Read the statistics from the file. Return a dict with keys "timers", "counters" """
    res = {}
    try:
        with open(fname) as f:
            res["timers"]   = readTimers(f)
            res["counters"] = readCounters(f)
            return res
    except (OSError, IOError):
        print "Cannot open " + fname
        return None

def usefulValues(l):
    """I.e. values which are neither null nor zero"""
    return [p and q for (p,q) in zip (pd.notnull(l), l != 0.0)]

def uselessValues(l):
    """I.e. values which are null or zero"""
    return [not p for p in usefulValues(l)]

interestingStats = ("counters", "timers")
statProperties   = {"counters" : ("Count", "Counter Statistics"),
                    "timers"   : ("Time (ticks)", "Timer Statistics")
                   }

def drawChart(data, kind, filebase):
    """Draw a summary bar chart for the requested data frame into the specified file"""
    data["Mean"].plot(kind="bar", logy=True, grid=True, colormap="GnBu",
                      yerr=data["SD"], ecolor="black")
    plt.xlabel("OMP Constructs")
    plt.ylabel(statProperties[kind][0])
    plt.title (statProperties[kind][1])
    plt.tight_layout()
    plt.savefig(filebase+"_"+kind)

def normalizeValues(data, countField, factor):
    """Normalize values into a rate by dividing them all by the given factor"""
    data[[k for k in data.keys() if k != countField]] /= factor


def setRadarFigure(titles):
    """Set the attributes for the radar plots"""
    fig = plt.figure(figsize=(9,9))
    rect = [0.1, 0.1, 0.8, 0.8]
    labels = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 10]
    matplotlib.rcParams.update({'font.size':13})
    theta = radar_factory(len(titles))
    ax = fig.add_axes(rect, projection='radar')
    ax.set_rgrids(labels)
    ax.set_varlabels(titles)
    ax.text(theta[2], 1, "Linear->Log", horizontalalignment='center', color='green', fontsize=18)
    return {'ax':ax, 'theta':theta}


def drawRadarChart(data, kind, filebase, params, color):
    """Draw the radar plots"""
    tmp_lin = data * 0
    tmp_log = data * 0
    for key in data.keys():
        if data[key] >= 1:
           tmp_log[key] = np.log10(data[key])
        else:
           tmp_lin[key] = (data[key])
    params['ax'].plot(params['theta'], tmp_log, color='b', label=filebase+"_"+kind+"_log")
    params['ax'].plot(params['theta'], tmp_lin, color='r', label=filebase+"_"+kind+"_linear")
    params['ax'].legend(loc='best', bbox_to_anchor=(1.4,1.2))
    params['ax'].set_rlim((0, np.ceil(max(tmp_log))))

def multiAppBarChartSettings(ax, plt, index, width, n, tmp, s):
    ax.set_yscale('log')
    ax.legend()
    ax.set_xticks(index + width * n / 2)
    ax.set_xticklabels(tmp[s]['Total'].keys(), rotation=50, horizontalalignment='right')
    plt.xlabel("OMP Constructs")
    plt.ylabel(statProperties[s][0])
    plt.title(statProperties[s][1])
    plt.tight_layout()

def derivedTimerStats(data):
    stats = {}
    for key in data.keys():
        if key == 'OMP_worker_thread_life':
            totalRuntime = data['OMP_worker_thread_life']
        elif key in ('FOR_static_iterations', 'OMP_PARALLEL_args',
                     'OMP_set_numthreads', 'FOR_dynamic_iterations'):
            break
        else:
            stats[key] = 100 * data[key] / totalRuntime
    return stats

def compPie(data):
    compKeys = {}
    nonCompKeys = {}
    for key in data.keys():
        if key in ('OMP_critical', 'OMP_single', 'OMP_serial',
                   'OMP_parallel', 'OMP_master', 'OMP_task_immediate',
                   'OMP_task_taskwait', 'OMP_task_taskyield', 'OMP_task_taskgroup',
                   'OMP_task_join_bar', 'OMP_task_plain_bar', 'OMP_task_taskyield'):
            compKeys[key] = data[key]
        else:
            nonCompKeys[key] = data[key]
    print "comp keys:", compKeys, "\n\n non comp keys:", nonCompKeys
    return [compKeys, nonCompKeys]

def drawMainPie(data, filebase, colors):
    sizes = [sum(data[0].values()), sum(data[1].values())]
    explode = [0,0]
    labels = ["Compute - " + "%.2f" % sizes[0], "Non Compute - " + "%.2f" % sizes[1]]
    patches = plt.pie(sizes, explode, colors=colors, startangle=90)
    plt.title("Time Division")
    plt.axis('equal')
    plt.legend(patches[0], labels, loc='best', bbox_to_anchor=(-0.1,1), fontsize=16)
    plt.savefig(filebase+"_main_pie", bbox_inches='tight')

def drawSubPie(data, tag, filebase, colors):
    explode = []
    labels = data.keys()
    sizes = data.values()
    total = sum(sizes)
    percent = []
    for i in range(len(sizes)):
        explode.append(0)
        percent.append(100 * sizes[i] / total)
        labels[i] = labels[i] + " - %.2f" % percent[i]
    patches = plt.pie(sizes, explode=explode, colors=colors, startangle=90)
    plt.title(tag+"(Percentage of Total:"+" %.2f" % (sum(data.values()))+")")
    plt.tight_layout()
    plt.axis('equal')
    plt.legend(patches[0], labels, loc='best', bbox_to_anchor=(-0.1,1), fontsize=16)
    plt.savefig(filebase+"_"+tag, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description='''This script takes a list
        of files containing each of which contain output from a stats-gathering
        enabled OpenMP runtime library.  Each stats file is read, parsed, and
        used to produce a summary of the statistics''')
    parser.add_argument('files', nargs='+',
        help='files to parse which contain stats-gathering output')
    command_args = parser.parse_args()
    colors = ['orange', 'b', 'r', 'yellowgreen', 'lightsage', 'lightpink',
              'green', 'purple', 'yellow', 'cyan', 'mediumturquoise',
              'olive']
    stats = {}
    matplotlib.rcParams.update({'font.size':22})
    for s in interestingStats:
        fig, ax = plt.subplots()
        width = 0.45
        n = 0
        index = 0

        for f in command_args.files:
            filebase = os.path.splitext(f)[0]
            tmp = readFile(f)
            data = tmp[s]['Total']
            """preventing repetition by removing rows similar to Total_OMP_work
                as Total_OMP_work['Total'] is same as OMP_work['Total']"""
            if s == 'counters':
                elapsedTime = tmp["timers"]["Mean"]["OMP_worker_thread_life"]
                normalizeValues(tmp["counters"], "SampleCount",
                    elapsedTime / 1.e9)
                """Plotting radar charts"""
                params = setRadarFigure(data.keys())
                chartType = "radar"
                drawRadarChart(data, s, filebase, params, colors[n])
                """radar Charts finish here"""
                plt.savefig(filebase+"_"+s+"_"+chartType, bbox_inches='tight')
            elif s == 'timers':
                print "overheads in "+filebase
                numThreads = tmp[s]['SampleCount']['Total_OMP_parallel']
                for key in data.keys():
                    if key[0:5] == 'Total':
                        del data[key]
                stats[filebase] = derivedTimerStats(data)
                dataSubSet = compPie(stats[filebase])
                drawMainPie(dataSubSet, filebase, colors)
                plt.figure(0)
                drawSubPie(dataSubSet[0], "Computational Time", filebase, colors)
                plt.figure(1)
                drawSubPie(dataSubSet[1], "Non Computational Time", filebase, colors)
                with open('derivedStats_{}.csv'.format(filebase), 'w') as f:
                    f.write('================={}====================\n'.format(filebase))
                    f.write(pd.DataFrame(stats[filebase].items()).to_csv()+'\n')
            n += 1
    plt.close()

if __name__ == "__main__":
    main()
