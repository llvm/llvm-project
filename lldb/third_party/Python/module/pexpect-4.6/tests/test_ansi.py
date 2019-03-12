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
from pexpect import ANSI
import unittest
from . import PexpectTestCase
import sys

PY3 = (sys.version_info[0] >= 3)

write_target = 'I\'ve got a ferret sticking up my nose.                           \n' +\
'(He\'s got a ferret sticking up his nose.)                        \n' +\
'How it got there I can\'t tell                                    \n' +\
'But now it\'s there it hurts like hell                            \n' +\
'And what is more it radically affects my sense of smell.         \n' +\
'(His sense of smell.)                                            '

write_text = 'I\'ve got a ferret sticking up my nose.\n' + \
'(He\'s got a ferret sticking up his nose.)\n' + \
'How it got there I can\'t tell\n' + \
'But now it\'s there it hurts like hell\n' + \
'And what is more it radically affects my sense of smell.\n' + \
'(His sense of smell.)\n' + \
'I can see a bare-bottomed mandril.\n' + \
'(Slyly eyeing his other nostril.)\n' + \
'If it jumps inside there too I really don\'t know what to do\n' + \
'I\'ll be the proud posessor of a kind of nasal zoo.\n' + \
'(A nasal zoo.)\n' + \
'I\'ve got a ferret sticking up my nose.\n' + \
'(And what is worst of all it constantly explodes.)\n' + \
'"Ferrets don\'t explode," you say\n' + \
'But it happened nine times yesterday\n' + \
'And I should know for each time I was standing in the way.\n' + \
'I\'ve got a ferret sticking up my nose.\n' + \
'(He\'s got a ferret sticking up his nose.)\n' + \
'How it got there I can\'t tell\n' + \
'But now it\'s there it hurts like hell\n' + \
'And what is more it radically affects my sense of smell.\n' + \
'(His sense of smell.)'

tetris_target='                           XX            XXXX    XX                             \n' +\
'                           XXXXXX    XXXXXXXX    XX                             \n' +\
'                           XXXXXX    XXXXXXXX    XX                             \n' +\
'                           XX  XX    XX  XXXX    XX                             \n' +\
'                           XXXXXX  XXXX  XXXX    XX                             \n' +\
'                           XXXXXXXXXX    XXXX    XX                             \n' +\
'                           XX  XXXXXX      XX    XX                             \n' +\
'                           XXXXXX          XX    XX                             \n' +\
'                           XXXX    XXXXXX  XX    XX                             \n' +\
'                           XXXXXX    XXXX  XX    XX                             \n' +\
'                           XX  XX    XXXX  XX    XX                             \n' +\
'                           XX  XX      XX  XX    XX                             \n' +\
'                           XX  XX    XXXX  XXXX  XX                             \n' +\
'                           XXXXXXXX  XXXX  XXXX  XX                             \n' +\
'                           XXXXXXXXXXXXXX  XXXXXXXX                             \n' +\
'                           XX    XXXXXXXX  XX    XX                             \n' +\
'                           XXXXXXXXXXXXXX  XX    XX                             \n' +\
'                           XX  XXXX    XXXXXX    XX                             \n' +\
'                           XXXXXX          XXXXXXXX                             \n' +\
'                           XXXXXXXXXX      XX    XX                             \n' +\
'                           XXXXXXXXXXXXXXXXXXXXXXXX                             \n' +\
'                                                                                \n' +\
'  J->LEFT  K->ROTATE  L->RIGHT  SPACE->DROP  P->PAUSE  Q->QUIT                  \n' +\
'                                                                                '

torture_target='+--------------------------------------------------------------------------------+\n' +\
'|a`opqrs`      This is the       `srqpo`a                                        |\n' +\
'|VT100 series Torture Test Demonstration.                                        |\n' +\
'|VT100 series Torture Test Demonstration.                                        |\n' +\
'|This is a normal line __________________________________________________y_      |\n' +\
'|This is a bold line (normal unless the Advanced Video Option is installed)      |\n' +\
'|This line is underlined _ "       "       "       "       "       "    _y_      |\n' +\
'|This is a blinking line _ "       "       "       "       "       "    _y_      |\n' +\
'|This is inverse video _ (underlined if no AVO and cursor is underline) _y_      |\n' +\
'|Normal gjpqy Underline   Blink   Underline+Blink gjpqy                          |\n' +\
'|Bold   gjpqy Underline   Blink   Underline+Blink gjpqy                          |\n' +\
'|Inverse      Underline   Blink   Underline+Blink                                |\n' +\
'|Bold+Inverse Underline   Blink   Underline+Blink                                |\n' +\
'|This is double width                                                            |\n' +\
'|This is double height                                                           |\n' +\
'|This is double height                                                           |\n' +\
'|_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ioy                                        |\n' +\
'|_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ioy                                        |\n' +\
'|_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ioy                                        |\n' +\
'|`abcdefghijklmnopqrstuvwxyz{|}~ lqwqk                                           |\n' +\
'|`abcdefghijklmnopqrstuvwxyz{|}~ tqnqu                                           |\n' +\
'|`abcdefghijklmnopqrstuvwxyz{|}~ tqnqu                                           |\n' +\
'|`abcdefghijklmnopqrstuvwxyz{|}~ mqvqj                                           |\n' +\
'|   This test created by Joe Smith, 8-May-85                                     |\n' +\
'|                                                                                |\n' +\
'+--------------------------------------------------------------------------------+\n'

class ansiTestCase (PexpectTestCase.PexpectTestCase):
    def test_write (self):
        s = ANSI.ANSI (6,65)
        s.fill('.')
        s.cursor_home()
        for c in write_text:
            s.write (c)
        assert str(s) == write_target

    def test_torturet (self):
        s = ANSI.ANSI (24,80)
        with open('torturet.vt') as f:
            sample_text = f.read()
        for c in sample_text:
            s.process (c)
        assert s.pretty() == torture_target, 'processed: \n' + s.pretty() + '\nexpected:\n' + torture_target

    def test_tetris (self):
        s = ANSI.ANSI (24,80)
        with open('tetris.data') as f:
            tetris_text = f.read()
        for c in tetris_text:
            s.process (c)
        assert str(s) == tetris_target

    def test_lines(self):
        s = ANSI.ANSI(5, 5)
        s.write('a'*6 + '\n')
        s.write('ab\bcd\n')
        s.write('ab\rcd\n')
        assert str(s) == ('aaaaa\n'
                          'a    \n'
                          'acd  \n'
                          'cd   \n'
                          '     ')

    def test_number_x(self):
        """Test the FSM state used to handle more than 2 numeric parameters."""
        class TestANSI(ANSI.ANSI):
            captured_memory = None
            def do_sgr(self, fsm):
                assert self.captured_memory is None
                self.captured_memory = fsm.memory

        s = TestANSI(1, 20)
        s.write('\x1b[0;1;32;45mtest')
        assert str(s) == ('test                ')
        assert s.captured_memory is not None
        assert s.captured_memory == [s, '0', '1', '32', '45']

    def test_fsm_memory(self):
        """Test the FSM stack/memory does not have numbers left on it
        after some sequences with numbers are passed in."""
        s = ANSI.ANSI(1, 20)
        s.write('\x1b[0;1;2;3m\x1b[4;5;6;7q\x1b[?8h\x1b[?9ltest')
        assert str(s) == ('test                ')
        assert s.state.memory == [s]

    def test_utf8_bytes(self):
        """Test that when bytes are passed in containing UTF-8 encoded
        characters, where the encoding of each character consists of
        multiple bytes, the characters are correctly decoded.
        Incremental decoding is also tested."""
        s = ANSI.ANSI(2, 10, encoding='utf-8')
        # This is the UTF-8 encoding of the UCS character "HOURGLASS"
        # followed by the UTF-8 encoding of the UCS character
        # "KEYBOARD".  These characters can't be encoded in cp437 or
        # latin-1.  The "KEYBOARD" character is split into two
        # separate writes.
        s.write(b'\xe2\x8c\x9b')
        s.write(b'\xe2\x8c')
        s.write(b'\xa8')
        if PY3:
            assert str(s) == u'\u231b\u2328        \n          '
        else:
            assert unicode(s) == u'\u231b\u2328        \n          '
            assert str(s) == b'\xe2\x8c\x9b\xe2\x8c\xa8        \n          '
        assert s.dump() == u'\u231b\u2328                  '
        assert s.pretty() == u'+----------+\n|\u231b\u2328        |\n|          |\n+----------+\n'
        assert s.get_abs(1, 1) == u'\u231b'
        assert s.get_region(1, 1, 1, 5) == [u'\u231b\u2328   ']

    def test_unicode(self):
        """Test passing in of a unicode string."""
        s = ANSI.ANSI(2, 10, encoding="utf-8")
        s.write(u'\u231b\u2328')
        if PY3:
            assert str(s) == u'\u231b\u2328        \n          '
        else:
            assert unicode(s) == u'\u231b\u2328        \n          '
            assert str(s) == b'\xe2\x8c\x9b\xe2\x8c\xa8        \n          '
        assert s.dump() == u'\u231b\u2328                  '
        assert s.pretty() == u'+----------+\n|\u231b\u2328        |\n|          |\n+----------+\n'
        assert s.get_abs(1, 1) == u'\u231b'
        assert s.get_region(1, 1, 1, 5) == [u'\u231b\u2328   ']

    def test_decode_error(self):
        """Test that default handling of decode errors replaces the
        invalid characters."""
        s = ANSI.ANSI(2, 10, encoding="ascii")
        s.write(b'\xff') # a non-ASCII character
        # In unicode, the non-ASCII character is replaced with
        # REPLACEMENT CHARACTER.
        if PY3:
            assert str(s) == u'\ufffd         \n          '
        else:
            assert unicode(s) == u'\ufffd         \n          '
            assert str(s) == b'?         \n          '
        assert s.dump() == u'\ufffd                   '
        assert s.pretty() == u'+----------+\n|\ufffd         |\n|          |\n+----------+\n'
        assert s.get_abs(1, 1) == u'\ufffd'
        assert s.get_region(1, 1, 1, 5) == [u'\ufffd    ']


if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(ansiTestCase,'test')

