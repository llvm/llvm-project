#!/usr/bin/env python

import sys
import subprocess
import traceback
import json

data = json.load(sys.stdin)
testfile = sys.argv[1]

prefix = "CHECK: "

fails = 0
passes = 0
with open(testfile) as testfh:
    lineno = 0
    for line in iter(testfh.readline, ""):
        lineno += 1
        line = line.rstrip("\r\n")
        try:
            prefix_pos = line.index(prefix)
        except ValueError:
            continue
        check_expr = line[prefix_pos + len(prefix) :]

        try:
            exception = None
            result = eval(check_expr, {"data": data})
        except Exception:
            result = False
            exception = traceback.format_exc().splitlines()[-1]

        if exception is not None:
            sys.stderr.write(
                "{file}:{line:d}: check threw exception: {expr}\n"
                "{file}:{line:d}: exception was: {exception}\n".format(
                    file=testfile, line=lineno, expr=check_expr, exception=exception
                )
            )
            fails += 1
        elif not result:
            sys.stderr.write(
                "{file}:{line:d}: check returned False: {expr}\n".format(
                    file=testfile, line=lineno, expr=check_expr
                )
            )
            fails += 1
        else:
            passes += 1

if fails != 0:
    sys.exit("{} checks failed".format(fails))
else:
    sys.stdout.write("{} checks passed\n".format(passes))
