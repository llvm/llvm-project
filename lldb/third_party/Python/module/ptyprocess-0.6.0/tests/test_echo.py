import time
import unittest
from ptyprocess.ptyprocess import _is_solaris
from ptyprocess import PtyProcess

class PtyEchoTestCase(unittest.TestCase):

    def _read_until_eof(self, proc):
        """Read away all output on ``proc`` until EOF."""
        while True:
            try:
                proc.read()
            except EOFError:
                return

    @unittest.skipIf(_is_solaris, "waitnoecho cannot be called on this platform.")
    def test_waitnoecho_forever(self):
        """Ensure waitnoecho() with no timeout will return when echo=False."""
        cat = PtyProcess.spawn(['cat'], echo=False)
        assert cat.waitnoecho() == True
        assert cat.echo == False
        assert cat.getecho() == False
        cat.sendeof()
        self._read_until_eof(cat)
        assert cat.wait() == 0

    @unittest.skipIf(_is_solaris, "waitnoecho cannot be called on this platform.")
    def test_waitnoecho_timeout(self):
        """Ensure waitnoecho() with timeout will return when using stty to unset echo."""
        cat = PtyProcess.spawn(['cat'], echo=True)
        assert cat.waitnoecho(timeout=1) == False
        assert cat.echo == True
        assert cat.getecho() == True
        cat.sendeof()
        self._read_until_eof(cat)
        assert cat.wait() == 0
