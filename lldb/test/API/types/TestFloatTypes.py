"""
Test that variables of floating point types are displayed correctly.
"""



import AbstractBase

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FloatTypesTestCase(AbstractBase.GenericTester):

    def test_float_type(self):
        """Test that float-type variables are displayed correctly."""
        self.build_and_run('float.cpp', set(['float']))

    @skipUnlessDarwin
    def test_float_type_from_block(self):
        """Test that float-type variables are displayed correctly from a block."""
        self.build_and_run('float.cpp', set(['float']), bc=True)
