"""Test module to collect test executable hashsum."""
from litsupport import testplan
import hashlib
import logging
import platform


def compute(context):
    if hasattr(context, "executable_hash"):
        return
    executable = context.executable
    try:
        # AIX, z/OS, and Darwin's and Solaris' "strip" don't support these arguments.
        if platform.system() != 'OS/390' and platform.system() != 'AIX' and platform.system() != "Darwin" and platform.system() != "SunOS":
            stripped_executable = executable + ".stripped"
            testplan.check_call(
                [
                    context.config.strip_tool,
                    "--remove-section=.comment",
                    "--remove-section='.note*'",
                    "-o",
                    stripped_executable,
                    executable,
                ]
            )
            executable = stripped_executable

        h = hashlib.md5()
        h.update(open(executable, "rb").read())
        context.executable_hash = h.hexdigest()
    except Exception:
        logging.info("Could not calculate hash for %s" % executable)
        context.executable_hash = ""


def same_as_previous(context):
    """Check whether hash has changed compared to the results in
    config.previous_results."""
    previous_results = context.config.previous_results
    testname = context.test.getFullName()
    executable_hash = context.executable_hash
    if previous_results and "tests" in previous_results:
        for test in previous_results["tests"]:
            if "name" not in test or test["name"] != testname:
                continue
            if "metrics" not in test:
                continue
            metrics = test["metrics"]
            return "hash" in metrics and metrics["hash"] == executable_hash
    return False


def _getHash(context):
    compute(context)
    return {"hash": context.executable_hash}


def mutatePlan(context, plan):
    plan.metric_collectors.append(_getHash)
