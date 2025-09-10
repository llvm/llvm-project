"""Test Module to perform an extra execution of the benchmark in the linux
perf tool."""
from litsupport import shellcommand
from litsupport import testplan
from litsupport.modules import run_under


def _mutateCommandLine(context, commandline):
    context.profilefile = context.tmpBase + ".perf_data"
    # Storing profile file in context allows other modules to be aware of it.
    cmd = shellcommand.parse(commandline)
    cmd.wrap(
        "perf",
        [
            "record",
            "-e",
            context.config.perf_profile_events,
            "-o",
            context.profilefile,
            "--",
        ],
    )
    if cmd.stdout is None:
        cmd.stdout = "/dev/null"
    else:
        cmd.stdout += ".perfrecord"
    if cmd.stderr is None:
        cmd.stderr = "/dev/null"
    else:
        cmd.stderr += ".perfrecord"
    return cmd.toCommandline()


def mutatePlan(context, plan):
    script = context.parsed_runscript
    if context.config.run_under:
        script = testplan.mutateScript(context, script, run_under.mutateCommandLine)
    script = testplan.mutateScript(context, script, _mutateCommandLine)
    plan.profilescript += script
    plan.metric_collectors.append(lambda context: {"profile": context.profilefile})
