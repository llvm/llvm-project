try:
    import asyncio
except ImportError:
    asyncio = None

import gc
import sys
import unittest

import pexpect
from .PexpectTestCase import PexpectTestCase

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

@unittest.skipIf(asyncio is None, "Requires asyncio")
class AsyncTests(PexpectTestCase):
    def test_simple_expect(self):
        p = pexpect.spawn('cat')
        p.sendline('Hello asyncio')
        coro = p.expect(['Hello', pexpect.EOF] , async_=True)
        assert run(coro) == 0
        print('Done')

    def test_timeout(self):
        p = pexpect.spawn('cat')
        coro = p.expect('foo', timeout=1, async_=True)
        with self.assertRaises(pexpect.TIMEOUT):
            run(coro)
        
        p = pexpect.spawn('cat')
        coro = p.expect(['foo', pexpect.TIMEOUT], timeout=1, async_=True)
        assert run(coro) == 1

    def test_eof(self):
        p = pexpect.spawn('cat')
        p.sendline('Hi')
        coro = p.expect(pexpect.EOF, async_=True)
        p.sendeof()
        assert run(coro) == 0

        p = pexpect.spawn('cat')
        p.sendeof()
        coro = p.expect('Blah', async_=True)
        with self.assertRaises(pexpect.EOF):
            run(coro)

    def test_expect_exact(self):
        p = pexpect.spawn('%s list100.py' % sys.executable)
        assert run(p.expect_exact(b'5', async_=True)) == 0
        assert run(p.expect_exact(['wpeok', b'11'], async_=True)) == 1
        assert run(p.expect_exact([b'foo', pexpect.EOF], async_=True)) == 1

    def test_async_utf8(self):
        p = pexpect.spawn('%s list100.py' % sys.executable, encoding='utf8')
        assert run(p.expect_exact(u'5', async_=True)) == 0
        assert run(p.expect_exact([u'wpeok', u'11'], async_=True)) == 1
        assert run(p.expect_exact([u'foo', pexpect.EOF], async_=True)) == 1

    def test_async_and_gc(self):
        p = pexpect.spawn('%s sleep_for.py 1' % sys.executable, encoding='utf8')
        assert run(p.expect_exact(u'READY', async_=True)) == 0
        gc.collect()
        assert run(p.expect_exact(u'END', async_=True)) == 0

    def test_async_and_sync(self):
        p = pexpect.spawn('echo 1234', encoding='utf8', maxread=1)
        assert run(p.expect_exact(u'1', async_=True)) == 0
        assert p.expect_exact(u'2') == 0
        assert run(p.expect_exact(u'3', async_=True)) == 0
