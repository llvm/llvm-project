"""Test module to run benchmarks in a wrapper application. This is typically
used to prefix the benchmark command with simulator commands."""
from litsupport import shellcommand
from litsupport import testplan


def mutateCommandLine(context, commandline):
    cmd = shellcommand.parse(commandline)
    run_under_cmd = shellcommand.parse(context.config.run_under)

    if (
        run_under_cmd.stdin is not None
        or run_under_cmd.stdout is not None
        or run_under_cmd.stderr is not None
        or run_under_cmd.workdir is not None
        or run_under_cmd.envvars
    ):
        raise Exception("invalid run_under argument!")

    cmd.wrap(run_under_cmd.executable, run_under_cmd.arguments)

    return cmd.toCommandline()


def mutatePlan(context, plan):
    run_under = context.config.run_under
    if run_under:
        plan.runscript = testplan.mutateScript(
            context, plan.runscript, mutateCommandLine
        )
