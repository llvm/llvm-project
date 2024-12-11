import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBasicTypes(TestBase):

    mydir = TestBase.compute_mydir(__file__)


    def test(self):
        self.build()

        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        for t, v in [("char", "arr"),
                     ("int", "arri"),
                     ("float", "arrf"),
                     ("double", "arrd"),
                     ("A", "arrA")]:
            expected_c_str = ''
            if t == 'char':
                # When printing a char arr[] we print the C string that is pointed to
                expected_c_str = r' *"hello world" *'
            self.expect("expr i_"+v,
                        patterns = ['\('+t+' \*__indexable\) \$[0-9]+ = \(ptr: 0x[a-f0-9]+, upper bound: 0x[a-f0-9]+\)' + expected_c_str + '$'])
            self.expect("expr bi_"+v,
                        patterns = ['\('+t+' \*__bidi_indexable\) \$[0-9]+ = \(ptr: 0x[a-f0-9]+, bounds: 0x[a-f0-9]+..0x[a-f0-9]+\)' + expected_c_str + '$'])

            biArg = self.frame().FindVariable("bi_"+v)
            self.assertTrue(bool(re.match('0x[a-f0-9]+',biArg.GetValue())))

        lldbutil.continue_to_breakpoint(self.process, bkpt);

        self.expect("expr arrp",
            patterns = ['\(int \(\*__bidi_indexable\)\[8\]\) \$[0-9]+ = \(ptr: 0x[a-f0-9]+, bounds: 0x[a-f0-9]+..0x[a-f0-9]+\)$'])

        arrp = self.frame().FindVariable("arrp")
        self.assertTrue(bool(re.match('0x[a-f0-9]+',arrp.GetValue())))

        self.expect( "expr callback",
            patterns = ['\(int \(\*\)\(int \*__bidi_indexable\)\) \$[0-9]+ = 0x[a-f0-9]+'])

