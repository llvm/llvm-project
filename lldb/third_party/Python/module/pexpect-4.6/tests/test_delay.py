# -*- coding: utf-8 -*-

from . import PexpectTestCase
import pexpect


class TestCaseDelay(PexpectTestCase.PexpectTestCase):
    """
    Tests for various delay attributes.
    """
    def test_delaybeforesend(self):
        """
        Test various values for delaybeforesend.
        """
        p = pexpect.spawn("cat")

        p.delaybeforesend = 1
        p.sendline("line 1")
        p.expect("line 1")

        p.delaybeforesend = 0.0
        p.sendline("line 2")
        p.expect("line 2")

        p.delaybeforesend = None
        p.sendline("line 3")
        p.expect("line 3")

    def test_delayafterread(self):
        """
        Test various values for delayafterread.
        """
        p = pexpect.spawn("cat")

        p.delayafterread = 1
        p.sendline("line 1")
        p.expect("line 1")

        p.delayafterread = 0.0
        p.sendline("line 2")
        p.expect("line 2")

        p.delayafterread = None
        p.sendline("line 3")
        p.expect("line 3")
