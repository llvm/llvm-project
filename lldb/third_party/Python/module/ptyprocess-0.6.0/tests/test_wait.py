""" Test cases for PtyProcess.wait method. """
import time
import unittest
from ptyprocess import PtyProcess


class TestWaitAfterTermination(unittest.TestCase):
    """Various test cases for PtyProcess.wait()"""

    def test_wait_true_shortproc(self):
        """Ensure correct (True) wait status for short-lived processes."""
        child = PtyProcess.spawn(['true'])
        # Wait so we're reasonable sure /bin/true has terminated
        time.sleep(0.2)
        self.assertEqual(child.wait(), 0)

    def test_wait_false_shortproc(self):
        """Ensure correct (False) wait status for short-lived processes."""
        child = PtyProcess.spawn(['false'])
        # Wait so we're reasonable sure /bin/false has terminated
        time.sleep(0.2)
        self.assertNotEqual(child.wait(), 0)

    def test_wait_twice_longproc(self):
        """Ensure correct wait status when called twice."""
        # previous versions of ptyprocess raises PtyProcessError when
        # wait was called more than once with "Cannot wait for dead child
        # process.".  No longer true since v0.5.
        child = PtyProcess.spawn(['sleep', '1'])
        # this call to wait() will block for 1s
        for count in range(2):
            self.assertEqual(child.wait(), 0, count)
