import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftCCompatEnum(TestBase):
    @skipEmbeddedSwift
    @swiftTest
    def test(self):
        self.build()

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"))

        v = thread.GetSelectedFrame().FindVariable("v")
        self.assertTrue(v.IsValid(), "v not found in frame")

        # We can't actually inspect the enums because of 
        # rdar://177569037 (lldb can't show cases of an @objc or @c enum)
        # so we have to resort to reading the logs instead and make 
        # sure the fallback wasn't triggered.
        import io
        with io.open(log, "r", encoding="utf-8") as f:
            log_contents = f.read()

        def fired_fallback(type_name):
            for line in log_contents.splitlines():
                if "had to engage SwiftASTContext fallback" in line and \
                   type_name in line:
                    return line
            return None

        objc_fallback = fired_fallback("TheObjCEnum")
        self.assertIsNone(
            objc_fallback,
            "TypeSystemSwiftTypeRef fell through to AST-context fallback "
            "for the @objc enum: " + (objc_fallback or ""))

        c_fallback = fired_fallback("TheCEnum")
        self.assertIsNone(
            c_fallback,
            "TypeSystemSwiftTypeRef fell through to AST-context fallback "
            "for the @c enum: " + (c_fallback or ""))
