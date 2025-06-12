"""
Test that running Swift expressions in the
`memory find` command works.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftCommandMemoryFind(TestBase):
    def memory_find(self, name: str, expr: str, target):
        var = target.FindGlobalVariables(name, 1)
        self.assertEqual(len(var), 1)
        addr = var[0].AddressOf()
        self.assertTrue(addr)
        addr = addr.GetValueAsUnsigned()
        self.expect(f'memory find -e "{expr}" {hex(addr)} {hex(addr + 8)}',
                    substrs=["data found at location"])

    @swiftTest
    def test(self):
        self.build()
        target, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, 'Break', lldb.SBFileSpec('main.swift'))

        self.memory_find('elem1', 'elem1', target)
        self.memory_find('elem1', '130 + 7', target)

        self.memory_find('elem2', 'elem2', target)
        self.memory_find('elem2', '-42', target)
