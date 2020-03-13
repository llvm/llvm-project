"""
Test reproducer attach.
"""

import lldb
import tempfile
from lldbsuite.test import lldbtest_config
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CreateAfterAttachTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfFreeBSD
    @skipIfNetBSD
    @skipIfWindows
    @skipIfRemote
    @skipIfiOSSimulator
    def test_create_after_attach_with_fork(self):
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
        self.addTearDownHook(self.cleanupSubprocesses)

        inferior = self.spawnSubprocess(self.getBuildArtifact(exe), [token])
        pid = inferior.pid

        lldbutil.wait_for_file_on_target(self, token)

        # Use Popen because pexpect is overkill and spawnSubprocess is
        # asynchronous.
        capture = subprocess.Popen([
            lldbtest_config.lldbExec, '-b', '--capture', '--capture-path',
            reproducer, '-o', 'proc att -n {}'.format(exe), '-o',
            'reproducer generate'
        ],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        outs, errs = capture.communicate()
        self.assertIn('Process {} stopped'.format(pid), outs)
        self.assertIn('Reproducer written', outs)

        # Check that replay works.
        replay = subprocess.Popen(
            [lldbtest_config.lldbExec, '-replay', reproducer],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        outs, errs = replay.communicate()
        self.assertIn('Process {} stopped'.format(pid), outs)

        # We can dump the reproducer in the current context.
        self.expect('reproducer dump -f {} -p process'.format(reproducer),
                    substrs=['pid = {}'.format(pid), 'name = {}'.format(exe)])
