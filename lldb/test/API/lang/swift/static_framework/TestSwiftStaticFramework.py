import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftStaticFramework(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIf(oslist=no_match(["macosx"]))
    def test_static_framework(self):
        """Make sure LLDB doesn't attempt to import static frameworks"""
        n = 10
        for i in range(n):
            with open(self.getBuildArtifact("a%d.swift"%i), "w") as f:
                f.write("public struct A%d { public init() {} }\n"%i)
        src = self.getBuildArtifact("main.swift")
        with open(src, "w") as f:
            f.write("func use<T>(_ t: T) {}\n")
            for i in range(n):
                f.write("import a%d\n"%i)
            for i in range(n):
                f.write("let v%d = A%d()\n"%(i,i))
                f.write("use(v%d)\n"%(i))
            f.write('print("break here")\n')
        self.build(dictionary={"N": "%d"%(n-1)})
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec(src))

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        # Run the expression evaluator to trigger import of all static
        # frameworks.  The expression imports a new dynamic framework
        # as a sanity check to make sure this still works.
        self.expect("expression -- import Dylib")
        import_a0 = 0
        load_a0 = 0
        load_dylib = 0
        import io
        with open(log, "r", encoding='utf-8') as logfile:
            for line in logfile:
                if 'Loading linked framework "Dylib"' in line:
                    load_dylib += 1
                elif "Importing module a0" in line:
                    import_a0 += 1
                elif 'Loading linked framework "a0"' in line:
                    load_a0 += 1

        self.assertGreater(import_a0, 0, "sanity check failed")
        self.assertEqual(load_a0, 0, "attempted to import a0 again")
        self.assertEqual(load_dylib, 1, "failed to import dylib")
