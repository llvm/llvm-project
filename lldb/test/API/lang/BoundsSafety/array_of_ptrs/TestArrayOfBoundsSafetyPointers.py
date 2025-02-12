import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestArrayOfBoundsSafetyPointers(TestBase):
    def __run(self, build_dict):
        self.build(dictionary = build_dict)

        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, "// break here 1", lldb.SBFileSpec("main.c"))

        zero_init_pattern = '\(out-of-bounds ptr: 0x000000000000..0x000000000004, bounds: 0x000000000000..0x000000000000\)'
        array_patterns = [f'\[0\] = {zero_init_pattern}',
                          f'\[1\] = {zero_init_pattern}']

        self.expect("expr array_of_bounds_safety_pointers", patterns = array_patterns)
        self.expect("frame variable array_of_bounds_safety_pointers", patterns = array_patterns)

        self.expect("expr array_of_bounds_safety_pointers[0]", patterns = [zero_init_pattern])
        self.expect("frame variable array_of_bounds_safety_pointers[0]", patterns = [zero_init_pattern])

        self.expect("expr array_of_bounds_safety_pointers[1]", patterns = [zero_init_pattern])
        self.expect("frame variable array_of_bounds_safety_pointers[1]", patterns = [zero_init_pattern])

        lldbutil.continue_to_source_breakpoint(self, self.process, "// break here 2", lldb.SBFileSpec("main.c"))

        initialized_pattern = '\(ptr: (0x[0-9a-f]*), bounds: \\1..0x[0-9a-f]*\)'
        array_patterns = [f'\[0\] = {initialized_pattern}',
                          f'\[1\] = {zero_init_pattern}']

        self.expect("expr array_of_bounds_safety_pointers", patterns = array_patterns)
        self.expect("frame variable array_of_bounds_safety_pointers", patterns = array_patterns)

        self.expect("expr array_of_bounds_safety_pointers[0]", patterns = [initialized_pattern])
        self.expect("frame variable array_of_bounds_safety_pointers[0]", patterns = [initialized_pattern])

        self.expect("expr array_of_bounds_safety_pointers[1]", patterns = [zero_init_pattern])
        self.expect("frame variable array_of_bounds_safety_pointers[1]", patterns = [zero_init_pattern])

    def test_optimized(self):
        build_dict=dict(CFLAGS_EXTRAS="-O2 -Xclang -fbounds-safety")
        self.__run(build_dict)

    def test_unoptimized(self):
        build_dict=dict(CFLAGS_EXTRAS="-Xclang -fbounds-safety")
        self.__run(build_dict)
