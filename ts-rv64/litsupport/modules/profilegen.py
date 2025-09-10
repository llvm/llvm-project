"""Test module that runs llvm-profdata merge after executing the benchmark."""
from litsupport import shellcommand
from litsupport import testplan


def _mutateCommandline(context, commandline):
    """Adjust runscript to set a different value to the LLVM_PROFILE_FILE
    environment variable for each execution."""
    profilefile = context.tmpBase + ".profraw"
    context.profilefiles.append(profilefile)
    cmd = shellcommand.parse(commandline)
    cmd.envvars.update({"LLVM_PROFILE_FILE": profilefile})
    return cmd.toCommandline()


def _mutateScript(context, script):
    return testplan.mutateScript(context, script, _mutateCommandline)


def mutatePlan(context, plan):
    context.profilefiles = []
    # Adjust run steps to set LLVM_PROFILE_FILE environment variable.
    plan.runscript = _mutateScript(context, plan.runscript)
    plan.profile_files += context.profilefiles

    # Run profdata merge at the end
    profdatafile = context.executable + ".profdata"
    args = ["merge", "-output=%s" % profdatafile] + context.profilefiles
    mergecmd = shellcommand.ShellCommand(context.config.llvm_profdata, args)
    plan.profilescript += [mergecmd.toCommandline()]
