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

import sys

from pexpect import screen
import unittest
from . import PexpectTestCase

PY3 = (sys.version_info[0] >= 3)

fill1_target='XXXXXXXXXX\n' + \
'XOOOOOOOOX\n' + \
'XO::::::OX\n' + \
'XO:oooo:OX\n' + \
'XO:o..o:OX\n' + \
'XO:o..o:OX\n' + \
'XO:oooo:OX\n' + \
'XO::::::OX\n' + \
'XOOOOOOOOX\n' + \
'XXXXXXXXXX'
fill2_target = 'XXXXXXXXXXX\n' + \
'XOOOOOOOOOX\n' + \
'XO:::::::OX\n' + \
'XO:ooooo:OX\n' + \
'XO:o...o:OX\n' + \
'XO:o.+.o:OX\n' + \
'XO:o...o:OX\n' + \
'XO:ooooo:OX\n' + \
'XO:::::::OX\n' + \
'XOOOOOOOOOX\n' + \
'XXXXXXXXXXX'
put_target = '\\.3.5.7.9/\n' + \
'.........2\n' + \
'3.........\n' + \
'.........4\n' + \
'5...\\/....\n' + \
'..../\\...6\n' + \
'7.........\n' + \
'.........8\n' + \
'9.........\n' + \
'/2.4.6.8.\\'
scroll_target = '\\.3.5.7.9/\n' + \
'\\.3.5.7.9/\n' + \
'\\.3.5.7.9/\n' + \
'\\.3.5.7.9/\n' + \
'5...\\/....\n' + \
'..../\\...6\n' + \
'/2.4.6.8.\\\n' + \
'/2.4.6.8.\\\n' + \
'/2.4.6.8.\\\n' + \
'/2.4.6.8.\\'
insert_target = 'ZXZZZZZZXZ\n' +\
'.........2\n' +\
'3.........\n' +\
'.........4\n' +\
'Z5...\\/...\n' +\
'..../Z\\...\n' +\
'7.........\n' +\
'.........8\n' +\
'9.........\n' +\
'ZZ/2.4.6ZZ'
get_region_target = ['......', '.\\/...', './\\...', '......']

unicode_box_unicode_result = u'\u2554\u2557\n\u255A\u255D'
unicode_box_pretty_result = u'''\
+--+
|\u2554\u2557|
|\u255A\u255D|
+--+
'''
unicode_box_ascii_bytes_result = b'??\n??'
unicode_box_cp437_bytes_result = b'\xc9\xbb\n\xc8\xbc'
unicode_box_utf8_bytes_result = b'\xe2\x95\x94\xe2\x95\x97\n\xe2\x95\x9a\xe2\x95\x9d'

class screenTestCase (PexpectTestCase.PexpectTestCase):
    def make_screen_with_put (self):
        s = screen.screen(10,10)
        s.fill ('.')
        for r in range (1,s.rows + 1):
            if r % 2:
                s.put_abs (r, 1, str(r))
            else:
                s.put_abs (r, s.cols, str(r))
        for c in range (1,s.cols + 1):
            if c % 2:
                s.put_abs (1, c, str(c))
            else:
                s.put_abs (s.rows, c, str(c))
        s.put_abs(1,1, '\\')
        s.put_abs(1,s.cols, '/')
        s.put_abs(s.rows,1,'/')
        s.put_abs(s.rows, s.cols, '\\')
        s.put_abs(5,5,'\\')
        s.put_abs(5,6,'/')
        s.put_abs(6,5,'/')
        s.put_abs(6,6,'\\')
        return s

    def test_fill (self):
        s = screen.screen (10,10)
        s.fill_region (10,1,1,10,'X')
        s.fill_region (2,2,9,9,'O')
        s.fill_region (8,8,3,3,':')
        s.fill_region (4,7,7,4,'o')
        s.fill_region (6,5,5,6,'.')
        assert str(s) == fill1_target

        s = screen.screen (11,11)
        s.fill_region (1,1,11,11,'X')
        s.fill_region (2,2,10,10,'O')
        s.fill_region (9,9,3,3,':')
        s.fill_region (4,8,8,4,'o')
        s.fill_region (7,5,5,7,'.')
        s.fill_region (6,6,6,6,'+')
        assert str(s) == fill2_target
    def test_put (self):
        s = self.make_screen_with_put()
        assert str(s) == put_target
    def test_get_region (self):
        s = self.make_screen_with_put()
        r = s.get_region (4,4,7,9)
        assert r == get_region_target

    def test_cursor_save (self):
        s = self.make_screen_with_put()
        s.cursor_home (5,5)
        c = s.get()
        s.cursor_save()
        s.cursor_home()
        s.cursor_forward()
        s.cursor_down()
        s.cursor_unsave()
        assert s.cur_r == 5 and s.cur_c == 5
        assert c == s.get()
    def test_scroll (self):
        s = self.make_screen_with_put()
        s.scroll_screen_rows (1,4)
        s.scroll_down(); s.scroll_down(); s.scroll_down()
        s.scroll_down(); s.scroll_down(); s.scroll_down()
        s.scroll_screen_rows (7,10)
        s.scroll_up(); s.scroll_up(); s.scroll_up()
        s.scroll_up(); s.scroll_up(); s.scroll_up()
        assert str(s) == scroll_target
    def test_insert (self):
        s = self.make_screen_with_put()
        s.insert_abs (10,1,'Z')
        s.insert_abs (1,1,'Z')
        s.insert_abs (1,1,'Z')
        s.insert_abs (1,1,'Z')
        s.insert_abs (1,1,'Z')
        s.insert_abs (1,1,'Z')
        s.insert_abs (10,1,'Z')
        s.insert_abs (1,1,'Z')
        s.insert_abs (1,1,'Z')
        s.insert_abs (5,1,'Z')
        s.insert_abs (6,6,'Z')
        s.cursor_home (1,1) # Also test relative insert.
        s.insert ('Z')
        s.insert ('Z')
        s.insert ('Z')
        s.insert ('Z')
        s.insert_abs (1,8,'X')
        s.insert_abs (1,2,'X')
        s.insert_abs (10,9,'Z')
        s.insert_abs (10,9,'Z')
        assert str(s) == insert_target

    def make_screen_with_box_unicode(self, *args, **kwargs):
        '''Creates a screen containing a box drawn using double-line
        line drawing characters. The characters are fed in as
        unicode. '''
        s = screen.screen (2,2,*args,**kwargs)
        s.put_abs (1,1,u'\u2554')
        s.put_abs (1,2,u'\u2557')
        s.put_abs (2,1,u'\u255A')
        s.put_abs (2,2,u'\u255D')
        return s

    def make_screen_with_box_cp437(self, *args, **kwargs):
        '''Creates a screen containing a box drawn using double-line
        line drawing characters. The characters are fed in as
        CP437. '''
        s = screen.screen (2,2,*args,**kwargs)
        s.put_abs (1,1,b'\xc9')
        s.put_abs (1,2,b'\xbb')
        s.put_abs (2,1,b'\xc8')
        s.put_abs (2,2,b'\xbc')
        return s

    def make_screen_with_box_utf8(self, *args, **kwargs):
        '''Creates a screen containing a box drawn using double-line
        line drawing characters. The characters are fed in as
        UTF-8. '''
        s = screen.screen (2,2,*args,**kwargs)
        s.put_abs (1,1,b'\xe2\x95\x94')
        s.put_abs (1,2,b'\xe2\x95\x97')
        s.put_abs (2,1,b'\xe2\x95\x9a')
        s.put_abs (2,2,b'\xe2\x95\x9d')
        return s

    def test_unicode_ascii (self):
        # With the default encoding set to ASCII, we should still be
        # able to feed in unicode strings and get them back out:
        s = self.make_screen_with_box_unicode('ascii')
        if PY3:
            assert str(s) == unicode_box_unicode_result
        else:
            assert unicode(s) == unicode_box_unicode_result
            # And we should still get something for Python 2 str(), though
            # it might not be very useful
            str(s)

        assert s.pretty() == unicode_box_pretty_result

    def test_decoding_errors(self):
        # With strict error handling, it should reject bytes it can't decode
        with self.assertRaises(UnicodeDecodeError):
            self.make_screen_with_box_cp437('ascii', 'strict')

        # replace should turn them into unicode replacement characters, U+FFFD
        s = self.make_screen_with_box_cp437('ascii', 'replace')
        expected = u'\ufffd\ufffd\n\ufffd\ufffd'
        if PY3:
            assert str(s) == expected
        else:
            assert unicode(s) == expected

    def test_unicode_cp437 (self):
        # Verify decoding from and re-encoding to CP437.
        s = self.make_screen_with_box_cp437('cp437','strict')
        if PY3:
            assert str(s) == unicode_box_unicode_result
        else:
            assert unicode(s) == unicode_box_unicode_result
            assert str(s) == unicode_box_cp437_bytes_result
        assert s.pretty() == unicode_box_pretty_result

    def test_unicode_utf8 (self):
        # Verify decoding from and re-encoding to UTF-8.
        s = self.make_screen_with_box_utf8('utf-8','strict')
        if PY3:
            assert str(s) == unicode_box_unicode_result
        else:
            assert unicode(s) == unicode_box_unicode_result
            assert str(s) == unicode_box_utf8_bytes_result
        assert s.pretty() == unicode_box_pretty_result

    def test_no_bytes(self):
        s = screen.screen(2, 2, encoding=None)
        s.put_abs(1, 1, u'A')
        s.put_abs(2, 2, u'D')

        with self.assertRaises(TypeError):
            s.put_abs(1, 2, b'B')

        if PY3:
            assert str(s) == u'A \n D'
        else:
            assert unicode(s) == u'A \n D'
            # This will still work if it's limited to ascii
            assert str(s) == b'A \n D'

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(screenTestCase,'test')


