"""
Test reproducer attach.
"""

import lldb
import tempfile
from lldbsuite.test import lldbtest_config
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ReproducerAttachTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfNetBSD
    @skipIfWindows
    @skipIfRemote
    @skipIfiOSSimulator
    @skipIfReproducer
    def test_reproducer_attach(self):
        """Test thread creation after process attach."""
        exe = '%s_%d' % (self.testMethodName, os.getpid())

        token = self.getBuildArtifact(exe + '.token')
        if os.path.exists(token):
            os.remove(token)

        reproducer = self.getBuildArtifact(exe + '.reproducer')
        if os.path.exists(reproducer):
            try:
                shutil.rmtree(reproducer)
            except OSError:
                pass

        self.build(dictionary={'EXE': exe})

        inferior = self.spawnSubprocess(self.getBuildArtifact(exe), [token])
        pid = inferior.pid

        lldbutil.wait_for_file_on_target(self, token)

        # Use Popen because pexpect is overkill and spawnSubprocess is
        # asynchronous.
        capture = subprocess.Popen([
            lldbtest_config.lldbExec, '-b', '--no-lldbinit', '--no-use-colors']
            + sum(map(lambda x: ['-O', x], self.setUpCommands()), [])
            + ['--capture', '--capture-path', reproducer,
            '-o', 'proc att -n {}'.format(exe), '-o', 'reproducer generate'
        ],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        outs, _ = capture.communicate()
        outs = outs.decode('utf-8')
        self.assertIn('Process {} stopped'.format(pid), outs)
        self.assertIn('Reproducer written', outs)

        # Check that replay works.
        replay = subprocess.Popen(
            [lldbtest_config.lldbExec, '-replay', reproducer],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        outs, _ = replay.communicate()
        outs = outs.decode('utf-8')
        self.assertIn('Process {} stopped'.format(pid), outs)

        # We can dump the reproducer in the current context.
        self.expect('reproducer dump -f {} -p process'.format(reproducer),
                    substrs=['pid = {}'.format(pid), 'name = {}'.format(exe)])
