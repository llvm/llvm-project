"""
Datastructures for test plans; Parsing of .test files; Executing test plans.
"""
import lit.Test
import lit.TestRunner
import logging
import os
import subprocess


class TestPlan(object):
    """Describes how to execute a benchmark and how to collect metrics.
    A script is a list of strings containing shell commands. The available
    scripts are: preparescript, runscript, verifyscript, profilescript,
    metricscripts and are executed in this order.
    metric_collectors contains a list of functions executed after the scripts
    finished."""

    def __init__(self):
        self.runscript = []
        self.verifyscript = []
        self.metricscripts = {}
        self.metric_collectors = []
        self.preparescript = []
        self.profile_files = []
        self.profilescript = []


def mutateScript(context, script, mutator):
    """Apply `mutator` function to every command in the `script` array of
    strings. The mutator function is called with `context` and the string to
    be mutated and must return the modified string. Sets `context.tmpBase`
    to a path unique to every command."""
    previous_tmpbase = context.tmpBase
    i = 0
    mutated_script = []
    for line in script:
        number = ""
        if len(script) > 1:
            number = "-%s" % (i,)
            i += 1
        context.tmpBase = previous_tmpbase + number

        mutated_line = mutator(context, line)
        mutated_script.append(mutated_line)
    return mutated_script


def _executeScript(context, script, scriptBaseName, useExternalSh=True):
    """Execute an array of strings with shellcommands (a script)."""
    if len(script) == 0:
        return "", "", 0, None

    if useExternalSh:
        execdir = None
        executeFunc = lit.TestRunner.executeScript
    else:
        execdir = os.getcwd()
        executeFunc = lit.TestRunner.executeScriptInternal

    logging.info("\n".join(script))
    res = executeFunc(
        context.test,
        context.litConfig,
        context.tmpBase + "_" + scriptBaseName,
        script,
        execdir,
    )
    # The executeScript() functions return lit.Test.Result in some error
    # conditions instead of the normal tuples. Having different return types is
    # really annoying so we transform it back to the usual tuple.
    if isinstance(res, lit.Test.Result):
        out = ""
        err = res.output
        exitCode = 1
        timeoutInfo = None
    else:
        out = res[0]
        err = res[1]
        exitCode = res[2]
        timeoutInfo = res[3]

    # Log script in test output
    context.result_output += "\n" + "\n".join(script)
    # In case of an exitCode != 0 also log stdout/stderr
    if exitCode != 0:
        context.result_output += "\n" + out
        context.result_output += "\n" + err

    logging.info(out)
    logging.info(err)
    if exitCode != 0:
        logging.info("ExitCode: %s" % exitCode)
    return (out, err, exitCode, timeoutInfo)


def _executePlan(context, plan):
    """Executes a test plan (a TestPlan object)."""
    # Execute PREPARE: part of the test.
    _, _, exitCode, _ = _executeScript(context, plan.preparescript, "prepare")
    if exitCode != 0:
        return lit.Test.FAIL

    # Execute RUN: part of the test.
    _, _, exitCode, _ = _executeScript(context, plan.runscript, "run")
    if exitCode != 0:
        return lit.Test.FAIL

    # Execute VERIFY: part of the test.
    _, _, exitCode, _ = _executeScript(context, plan.verifyscript, "verify")
    if exitCode != 0:
        # The question here is whether to still collects metrics if the
        # benchmark results are invalid. I choose to avoid getting potentially
        # broken metric values as well for a broken test.
        return lit.Test.FAIL

    # Execute additional profile gathering actions setup by testing modules.
    _, _, exitCode, _ = _executeScript(context, plan.profilescript, "profile")
    if exitCode != 0:
        logging.warning("Profile script '%s' failed", plan.profilescript)

    # Perform various metric extraction steps setup by testing modules.
    for metric_collector in plan.metric_collectors:
        try:
            additional_metrics = metric_collector(context)
            for metric, value in additional_metrics.items():
                litvalue = lit.Test.toMetricValue(value)
                context.result_metrics[metric] = litvalue
        except Exception as e:
            logging.error(
                "Could not collect metric with %s", metric_collector, exc_info=e
            )

    # Execute the METRIC: part of the test.
    for metric, metricscript in plan.metricscripts.items():
        out, err, exitCode, timeoutInfo = _executeScript(
            context, metricscript, "metric"
        )
        if exitCode != 0:
            logging.warning("Metric script for '%s' failed", metric)
            continue
        try:
            value = lit.Test.toMetricValue(float(out))
        except ValueError:
            logging.warning(
                "Metric reported for '%s' is not a float: '%s', treating as JSON", metric, out
            )
            value = lit.Test.JSONMetricValue(out)
        finally:
            context.result_metrics[metric] = value

    return lit.Test.PASS


def executePlanTestResult(context, testplan):
    """Convenience function to invoke _executePlan() and construct a
    lit.test.Result() object for the results."""
    context.result_output = ""
    context.result_metrics = {}
    context.micro_results = {}

    result_code = _executePlan(context, testplan)

    # Build test result object
    result = lit.Test.Result(result_code, context.result_output)
    for key, value in context.result_metrics.items():
        result.addMetric(key, value)
    for key, value in context.micro_results.items():
        result.addMicroResult(key, value)

    return result


def check_output(commandline, *aargs, **dargs):
    """Wrapper around subprocess.check_output that logs the command."""
    logging.info(" ".join(commandline))
    return subprocess.check_output(commandline, *aargs, **dargs)


def check_call(commandline, *aargs, **dargs):
    """Wrapper around subprocess.check_call that logs the command."""
    logging.info(" ".join(commandline))
    return subprocess.check_call(commandline, *aargs, **dargs)


def default_read_result_file(context, path):
    with open(path) as fd:
        return fd.read()


class TestContext:
    """This class is used to hold data used while constructing a testrun.
    For example this can be used by modules modifying the commandline with
    extra instrumentation/measurement wrappers to pass the filenames of the
    results to a final data collection step."""

    def __init__(self, test, litConfig, tmpDir, tmpBase):
        self.test = test
        self.config = test.config
        self.litConfig = litConfig
        self.tmpDir = tmpDir
        self.tmpBase = tmpBase
        self.read_result_file = default_read_result_file
