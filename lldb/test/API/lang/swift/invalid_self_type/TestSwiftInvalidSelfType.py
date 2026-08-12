import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftInvalidSelfType(TestBase):
    @swiftTest
    @skipEmbeddedSwift
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"))
        frame = thread.frames[0]

        # `self`'s static type is the archetype, but dynamic resolution
        # normally rescues it. Force that resolution to fail.
        self_val = frame.FindVariable("self", lldb.eNoDynamicValues)
        slot = self_val.GetLoadAddress()
        self.assertNotEqual(slot, lldb.LLDB_INVALID_ADDRESS,
                            "could not locate self's slot")
        err = lldb.SBError()
        instance = process.ReadPointerFromMemory(slot, err)
        self.assertSuccess(err, "could not read self's instance pointer")
        process.WriteMemory(instance, (0xdeadbeef0).to_bytes(8, "little"), err)
        self.assertSuccess(err, "failed to clobber isa")

        # Sanity check: with the isa clobbered, dynamic-type resolution can no
        # longer concretize `self`, so its raw type stays the archetype.
        self.expect("frame variable --raw -- self", substrs=["τ_0_0"])

        value = frame.EvaluateExpression("self.number")
        self.assertFalse(value.GetError().Success())
        error = value.GetError().GetCString()
        # We should not surface the misleading wrapper error ...
        self.assertNotIn("'mutating' is not valid on instance methods in classes",
                         error)
        # ... instead 'self' is treated as not in scope.
        self.assertIn("cannot find 'self' in scope", error)