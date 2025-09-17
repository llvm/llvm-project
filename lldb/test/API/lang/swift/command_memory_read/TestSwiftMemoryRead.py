import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @swiftTest
    def test_scalar_types(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.selected_frame

        child = frame.var("name")
        for t in ("String", "$sSSD"):
            self.expect(
                f"memory read -t {t} {child.load_addr}",
                substrs=[f'(String) 0x{child.load_addr:x} = "dirk"'],
            )

        child = frame.var("number")
        for t in ("Int", "$sSiD"):
            self.expect(
                f"memory read -t {t} {child.load_addr}",
                substrs=[f"(Int) 0x{child.load_addr:x} = 41"],
            )

        child = frame.var("fact")
        for t in ("Bool", "$sSbD"):
            self.expect(
                f"memory read -t {t} {child.load_addr}",
                substrs=[f"(Bool) 0x{child.load_addr:x} = true"],
            )

    @swiftTest
    @expectedFailureAll
    def test_generic_types(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.selected_frame

        child = frame.var("maybe")
        for t in ("UInt64?", "$ss6UInt64VSgD"):
            self.expect(
                f"memory read -t {t} {child.load_addr}",
                substrs=[f"(UInt64?) 0x{child.load_addr:x} = nil"],
            )

        child = frame.var("bytes")
        for t in ("[UInt8]", "$sSays5UInt8VGD"):
            self.expect(
                f"memory read -t {t} {child.load_addr}",
                substrs=[f"([UInt8]) 0x{child.load_addr:x} = [1, 2, 4, 8]"],
            )
