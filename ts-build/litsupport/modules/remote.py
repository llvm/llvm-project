"""Test module to execute a benchmark through ssh on a remote device.
This assumes all relevant directories and files are present on the remote
device (typically transferred via `ninja rsync`, or shared by NFS)."""
from litsupport import testplan
import logging
import os
import subprocess


def _wrap_command(context, command):
    escaped_command = command.replace("'", "'\\''")
    return "%s %s '%s'" % (
        context.config.remote_client,
        context.config.remote_host,
        escaped_command,
    )


def _mutateCommandline(context, commandline):
    return _wrap_command(context, commandline)


def _mutateScript(context, script):
    def mutate(context, command):
        return _mutateCommandline(context, command)

    return testplan.mutateScript(context, script, mutate)


def remote_read_result_file(context, path):
    assert os.path.isabs(path)
    command = _wrap_command(context, "cat '%s'" % path)
    logging.info("$ %s", command)
    return subprocess.check_output(command, shell=True)


def mutatePlan(context, plan):
    plan.preparescript = _mutateScript(context, plan.preparescript)
    # We need the temporary directory to exist on the remote as well.
    command = _wrap_command(context, "mkdir -p '%s'" % os.path.dirname(context.tmpBase))
    plan.preparescript.insert(0, command)
    plan.runscript = _mutateScript(context, plan.runscript)
    plan.verifyscript = _mutateScript(context, plan.verifyscript)
    for name, script in plan.metricscripts.items():
        plan.metricscripts[name] = _mutateScript(context, script)

    # Merging profile data should happen on the host because that is where
    # the toolchain resides, however we have to retrieve the profile data
    # from the device first, add commands for that to the profile script.
    for path in plan.profile_files:
        assert os.path.isabs(path)
        command = "scp %s:%s %s" % (context.config.remote_host, path, path)
        plan.profilescript.insert(0, command)

    assert context.read_result_file is testplan.default_read_result_file
    context.read_result_file = remote_read_result_file
