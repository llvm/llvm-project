"""test-suite/lit plugin to collect internal llvm json statistics.

This assumes the benchmarks were built with the -save-stats=obj flag."""
import json
import logging
import os
from collections import defaultdict


def _mergeStats(global_stats, statsfilename):
    try:
        f = open(statsfilename, "rt")
        stats = json.load(f)
    except Exception as e:
        logging.warning("Could not read '%s'", statsfilename, exc_info=e)
        return
    for name, value in stats.items():
        global_stats[name] += value


def _getStats(context):
    # We compile multiple benchmarks in the same directory in SingleSource
    # mode. Only look at compiletime files starting with the name of our test.
    prefix = ""
    if context.config.single_source:
        prefix = "%s." % os.path.basename(context.executable)

    stats = defaultdict(lambda: 0.0)
    dir = os.path.dirname(context.test.getFilePath())
    for path, subdirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".stats") and file.startswith(prefix):
                fullpath = os.path.join(path, file)
                _mergeStats(stats, fullpath)

    if len(stats) == 0:
        logging.warning("No stats for '%s'", context.test.getFullName())

    result = dict()
    for key, value in stats.items():
        result[key] = value
    return result


def mutatePlan(context, plan):
    plan.metric_collectors.append(_getStats)
