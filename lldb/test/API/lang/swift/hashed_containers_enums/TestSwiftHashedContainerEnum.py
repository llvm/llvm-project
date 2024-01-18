# TestSwiftHashedContainerEnum.py

"""
Test combinations of hashed swift containers with enums as keys/values
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftHashedContainerEnum(TestBase):
    @swiftTest
    def test_any_object_type(self):
        """Test combinations of hashed swift containers with enums"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, '// break here', lldb.SBFileSpec('main.swift'))

        self.expect(
            'frame variable -d run -- testA',
            ordered=False,
            substrs=[
                'key = c',
                'value = 1',
                'key = b',
                'value = 2'])
        self.expect(
            'expr -d run -- testA',
            ordered=False,
            substrs=[
                'key = c',
                'value = 1',
                'key = b',
                'value = 2'])

        self.expect(
            'frame variable -d run -- testB',
            ordered=False,
            substrs=[
                'key = "a", value = 1',
                'key = "b", value = 2'])
        self.expect(
            'expr -d run -- testB',
            ordered=False,
            substrs=[
                'key = "a", value = 1',
                'key = "b", value = 2'])

        self.expect(
            'frame variable -d run -- testC',
            ordered=False,
            substrs=['key = b', 'value = 2'])
        self.expect(
            'expr -d run -- testC',
            ordered=False,
            substrs=['key = b', 'value = 2'])

        self.expect(
            'frame variable -d run -- testD',
            substrs=['[0] = c'])
        self.expect(
            'expr -d run -- testD',
            substrs=['[0] = c'])
