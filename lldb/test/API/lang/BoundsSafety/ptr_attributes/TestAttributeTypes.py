import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBasicTypes(TestBase):


    def test(self):
        self.build()

        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        ptr_regex = '0x[0-9a-f]+'

        counted_by_pattern = f'\(int \*\) .* = \(ptr: {ptr_regex} counted_by: num_elements - 1\)'
        self.expect("frame var ptr_counted_by", patterns = [counted_by_pattern])
        self.expect("expr ptr_counted_by", patterns = [counted_by_pattern])

        sized_by_pattern = f'\(int \*\) .* \(ptr: {ptr_regex} sized_by: num_elements \* 4\)'
        self.expect("frame var ptr_sized_by", patterns = [sized_by_pattern])
        self.expect("expr ptr_sized_by", patterns = [sized_by_pattern])

        counted_by_or_null_pattern = f'\(int \*\) .* = \(ptr: {ptr_regex} counted_by_or_null: num_elements - 5\)'
        self.expect("frame var ptr_counted_by_or_null", patterns = [counted_by_or_null_pattern])
        self.expect("expr ptr_counted_by_or_null", patterns = [counted_by_or_null_pattern])

        sized_by_or_null_pattern = f'\(int \*\) .* \(ptr: {ptr_regex} sized_by_or_null: num_elements \* 2\)'
        self.expect("frame var ptr_sized_by_or_null", patterns = [sized_by_or_null_pattern])
        self.expect("expr ptr_sized_by_or_null", patterns = [sized_by_or_null_pattern])

        ended_by_pattern = f'\(int \*\) .* \(ptr: {ptr_regex} end_expr: end\)'
        self.expect("frame var ptr_ended_by", patterns = [ended_by_pattern])
        self.expect("expr ptr_ended_by", patterns = [ended_by_pattern])

        # Using "end" inside an attribute like `ended_by(end)` makes it become
        # a pointer with a "start expr".
        end_pattern = f'\(int \*\) .* \(ptr: {ptr_regex} start_expr: ptr_ended_by\)'
        self.expect("frame var end", patterns = [end_pattern])
        self.expect("expr end", patterns = [end_pattern])

