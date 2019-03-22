import os.path
import pprint
import subprocess
import sys

import transfer.protocol


class RsyncOverSsh(transfer.protocol.Protocol):

    def __init__(self, options, config):
        super(RsyncOverSsh, self).__init__(options, config)
        self.ssh_config = config.get_value("ssh")

    def build_rsync_command(self, transfer_spec, dry_run):
        dest_path = os.path.join(
            self.ssh_config["root_dir"],
            transfer_spec.dest_path)
        flags = "-avz"
        if dry_run:
            flags += "n"
        cmd = [
            "rsync",
            flags,
            "-e",
            "ssh -p {}".format(self.ssh_config["port"]),
            "--rsync-path",
            # The following command needs to know the right way to do
            # this on the dest platform - ensures the target dir exists.
            "mkdir -p {} && rsync".format(dest_path)
        ]

        # Add source dir exclusions
        if transfer_spec.exclude_paths:
            for exclude_path in transfer_spec.exclude_paths:
                cmd.append("--exclude")
                cmd.append(exclude_path)

        cmd.extend([
            "--delete",
            transfer_spec.source_path + "/",
            "{}@{}:{}".format(
                self.ssh_config["user"],
                self.ssh_config["dest_host"],
                dest_path)])
        return cmd

    def transfer(self, transfer_specs, dry_run):
        if self.options.verbose:
            printer = pprint.PrettyPrinter()
            for spec in transfer_specs:
                printer.pprint(spec)

        for spec in transfer_specs:
            cmd = self.build_rsync_command(spec, dry_run)
            if self.options.verbose:
                print("executing the following command:\n{}".format(cmd))
            result = subprocess.call(
                cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
            if result != 0:
                return result
