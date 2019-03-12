import os
import time
import select
import unittest
from ptyprocess.ptyprocess import which
from ptyprocess import PtyProcess, PtyProcessUnicode

class PtyTestCase(unittest.TestCase):
    def setUp(self):
        self.cmd = u'echo $ENV_KEY; exit 0\n'
        self.env = os.environ.copy()
        self.env_key = u'ENV_KEY'
        self.env_value = u'env_value'
        self.env[self.env_key] = self.env_value

    def _canread(self, fd, timeout=1):
        return fd in select.select([fd], [], [], timeout)[0]

    def _spawn_sh(self, ptyp, cmd, outp, env_value):
        # given,
        p = ptyp.spawn(['sh'], env=self.env)
        p.write(cmd)

        # exercise,
        while True:
            try:
                outp += p.read()
            except EOFError:
                break

        # verify, input is echo to output
        assert cmd.strip() in outp

        # result of echo $ENV_KEY in output
        assert env_value in outp

        # exit successfully (exit 0)
        assert p.wait() == 0


    def test_spawn_sh(self):
        outp = b''
        self._spawn_sh(PtyProcess, self.cmd.encode('ascii'),
                       outp, self.env_value.encode('ascii'))

    def test_spawn_sh_unicode(self):
        outp = u''
        self._spawn_sh(PtyProcessUnicode, self.cmd,
                       outp, self.env_value)

    def test_quick_spawn(self):
        """Spawn a very short-lived process."""
        # so far only reproducible on Solaris 11, spawning a process
        # that exits very quickly raised an exception at 'inst.setwinsize',
        # because the pty file descriptor was quickly lost after exec().
        PtyProcess.spawn(['true'])

    def _interactive_repl_unicode(self, echo):
        """Test Call and response with echo ON/OFF."""
        # given,
        bc = PtyProcessUnicode.spawn(['bc'], echo=echo)
        given_input = u'2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2\n'
        expected_output = u'40'

        # gnu-bc will display a long FSF banner on startup,
        # whereas bsd-bc (on FreeBSD, Solaris) display no
        # banner at all.  To ensure we've read up to our
        # current prompt, read until the response of '2^16' is found.
        time.sleep(1)

        bc.write(u'2^16\n')
        outp = u''
        while self._canread(bc.fd):
            outp += bc.read()
        assert u'65536' in outp

        # exercise,
        bc.write(given_input)

        while self._canread(bc.fd, timeout=2):
            outp += bc.read()

        # with echo ON, we should see our input.
        #
        # note: we cannot assert the reverse: on Solaris, FreeBSD,
        # and OSX, our input is echoed to output even with echo=False,
        # something to do with the non-gnu version of bc(1), perhaps.
        if echo:
            assert given_input.strip() in outp

        # we should most certainly see the result output.
        assert expected_output in outp

        # exercise sending EOF
        bc.sendeof()

        # validate EOF on read
        while True:
            try:
                bc.read()
            except EOFError:
                break

        # validate exit status,
        assert bc.wait() == 0

    @unittest.skipIf(which('bc') is None, "bc(1) not found on this server.")
    def test_interactive_repl_unicode_noecho(self):
        self._interactive_repl_unicode(echo=False)

    @unittest.skipIf(which('bc') is None, "bc(1) not found on this server.")
    def test_interactive_repl_unicode_echo(self):
        self._interactive_repl_unicode(echo=True)
