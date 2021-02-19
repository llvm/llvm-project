"""
Test if the reproducer correctly detects whether the file system is case sensitive.
"""

import lldb
import tempfile
from lldbsuite.test import lldbtest_config
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ReproducerFileSystemSensitivityTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfNetBSD
    @skipIfWindows
    @skipIfRemote
    @skipIfiOSSimulator
    @skipIfReproducer
    def test_reproducer_attach(self):
        # The reproducer output path. Note that this is on purpose a lower-case
        # file name. See the case-sensitivity check below.
        reproducer = self.getBuildArtifact('test.reproducer')
        try:
            shutil.rmtree(reproducer)
        except OSError:
            pass
        # Use Popen because pexpect is overkill and spawnSubprocess is
        # asynchronous.
        capture = subprocess.Popen([
            lldbtest_config.lldbExec, '-b', '--no-lldbinit', '--no-use-colors']
            + sum(map(lambda x: ['-O', x], self.setUpCommands()), [])
            + ['--capture', '--capture-path', reproducer,
            '-o', 'reproducer generate'
        ],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        outs, _ = capture.communicate()
        outs = outs.decode('utf-8')
        self.assertIn('Reproducer written', outs)

        # Read in the YAML file. We only care about a single value, so no
        # need to parse the full file.
        with open(os.path.join(reproducer, "files.yaml"), 'r') as file:
            files_yaml = file.read()

        # Detect the file system case sensitivity by checking if we can
        # find the reproducer path after converting it to upper case (the
        # file name is lower case before conversion, so this only works
        # on case insensitive file systems).
        case_sensitive = "false" if os.path.exists(reproducer.upper()) else "true"

        self.assertIn("'case-sensitive': '" + case_sensitive + "'", files_yaml)
