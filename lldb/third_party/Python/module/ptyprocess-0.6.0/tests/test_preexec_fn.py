#!/usr/bin/env python
'''
PEXPECT LICENSE

    This license is approved by the OSI and FSF as GPL-compatible.
        http://opensource.org/licenses/isc-license.txt

    Copyright (c) 2012, Noah Spurrier <noah@noah.org>
    PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE THIS SOFTWARE FOR ANY
    PURPOSE WITH OR WITHOUT FEE IS HEREBY GRANTED, PROVIDED THAT THE ABOVE
    COPYRIGHT NOTICE AND THIS PERMISSION NOTICE APPEAR IN ALL COPIES.
    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

'''
import unittest
import shutil
from ptyprocess import PtyProcess
import os
import tempfile

class PreexecFns(unittest.TestCase):
    def test_preexec(self):
        td = tempfile.mkdtemp()
        filepath = os.path.join(td, 'foo')
        def pef():
            with open(filepath, 'w') as f:
                f.write('bar')

        try:
            child = PtyProcess.spawn(['ls'], preexec_fn=pef)
            child.close()
            with open(filepath, 'r') as f:
                assert f.read() == 'bar'

        finally:
            shutil.rmtree(td)

    def test_preexec_error(self):
        def func():
            raise ValueError("Test error condition")

        try:
            child = PtyProcess.spawn(['ls'], preexec_fn=func)
            # If we get here then an error was not raised
            child.close()
            raise AssertionError("ValueError was not raised")
        except ValueError as err:
            if str(err) != "Test error condition":
                # Re-raise the original error to fail the test
                raise


