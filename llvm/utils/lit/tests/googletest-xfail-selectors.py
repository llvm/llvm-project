# RUN: %{python} %s

import json
import os
import subprocess
import sys
import tempfile

lit = os.path.join(os.path.dirname(__file__), '..', 'lit.py')
fixture = os.path.join(os.path.dirname(__file__), 'Inputs', 'googletest-format')
prefix = 'googletest-format :: DummySubDir/OneTest.py/'
failed = prefix + 'FirstTest/subTestB'
passed = prefix + 'FirstTest/subTestA'


def results(*args):
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, 'results.json')
        process = subprocess.run(
            [sys.executable, lit, '-j1', '--order=lexical', '-o', report, *args, fixture],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        # The fixture also contains an unresolved test, so failure is expected.
        assert process.returncode == 1, process.stdout + process.stderr
        with open(report) as stream:
            return {test['name']: test['code'] for test in json.load(stream)['tests']}


assert results('--xfail', failed)[failed] == 'XFAIL'
assert results('--xfail', passed)[passed] == 'XPASS'
relative = os.path.join('DummySubDir', 'OneTest.py', 'FirstTest', 'subTestB')
assert results('--xfail', relative)[failed] == 'XFAIL'
assert results('--xfail', failed, '--xfail-not', failed)[failed] == 'FAIL'
assert results('--xfail', passed, '--xfail-not', passed)[passed] == 'PASS'
parent = prefix + '0/6'
inherited = results('--xfail', parent)
assert inherited[failed] == 'XFAIL'
assert inherited[passed] == 'PASS'
assert results('--xfail', parent, '--xfail-not', failed)[failed] == 'FAIL'
assert results()[failed] == 'FAIL'
