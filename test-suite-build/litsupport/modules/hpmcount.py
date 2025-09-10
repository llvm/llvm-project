"""Test Module to perform an extra execution of the benchmark in the AIX
hpmcount tool."""
from litsupport import shellcommand
from litsupport import testplan
from litsupport.modules import run_under

import os


def _mutateCommandLine(context, commandline):
    context.profilefile = context.tmpBase + ".hpmcount_data"
    # Storing profile file in context allows other modules to be aware of it.
    cmd = shellcommand.parse(commandline)
    cmd.wrap("hpmcount", ["-o", context.profilefile])
    if cmd.stdout is None:
        cmd.stdout = os.devnull
    else:
        cmd.stdout += ".hpmcount"
    if cmd.stderr is None:
        cmd.stderr = os.devnull
    else:
        cmd.stderr += ".hpmcount"
    return cmd.toCommandline()


def mutatePlan(context, plan):
    script = context.parsed_runscript
    if context.config.run_under:
        script = testplan.mutateScript(context, script, run_under.mutateCommandLine)
    script = testplan.mutateScript(context, script, _mutateCommandLine)
    plan.profilescript += script
    plan.metric_collectors.append(lambda context: {"profile": context.profilefile})
