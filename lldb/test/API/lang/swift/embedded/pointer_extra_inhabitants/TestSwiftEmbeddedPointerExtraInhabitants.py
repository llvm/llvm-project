import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedPointerExtraInhabitants(TestBase):
    """
    Tests the extra inhabitant count LLDB reports for a not-nullable
    pointer-sized builtin in embedded Swift.

    A thin function pointer ("yyXf") exercises both halves of that: its
    mangling demangles to a function type, which has no decl context to look a
    type up in, and it is emitted as a DW_TAG_structure_type wrapping a "ptr"
    member rather than a base type.

    The count is observable through nested Optionals, because the Nth level of
    nesting needs an Nth extra inhabitant of the payload. With enough of them
    the whole nest stays pointer sized; with only one -- the Builtin.RawPointer
    answer -- every level past the first grows a discriminator byte.
    """

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_not_nullable_pointer_extra_inhabitants(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        frame = thread.GetSelectedFrame()

        # Every level is spare-bit encoded into the pointer itself, so all three
        # fields are 8 bytes and the struct is 3 * 8 = 24.
        fn_holder = frame.FindVariable("fnHolder")
        self.assertSuccess(fn_holder.GetError(), "fnHolder is available")
        self.assertEqual(fn_holder.GetType().GetByteSize(), 24)
        for name in ["a", "b", "c"]:
            field = fn_holder.GetChildMemberWithName(name)
            self.assertSuccess(field.GetError(), "fnHolder.%s is available" % name)
            self.assertEqual(
                field.GetType().GetByteSize(),
                8,
                "nested Optional of a thin function pointer stays pointer sized",
            )

        # The control: one extra inhabitant, so 8, then a byte per extra level.
        # 8 + 9 + (10 padded up to alignment 8, i.e. 16) = 34.
        raw_holder = frame.FindVariable("rawHolder")
        self.assertSuccess(raw_holder.GetError(), "rawHolder is available")
        self.assertEqual(raw_holder.GetType().GetByteSize(), 34)
        for name, size in [("a", 8), ("b", 9), ("c", 10)]:
            field = raw_holder.GetChildMemberWithName(name)
            self.assertSuccess(field.GetError(), "rawHolder.%s is available" % name)
            self.assertEqual(field.GetType().GetByteSize(), size)

        # If the function pointer is ever given RawPointer's count again,
        # FnHolder collapses onto RawHolder's layout.
        self.assertNotEqual(
            fn_holder.GetType().GetByteSize(),
            raw_holder.GetType().GetByteSize(),
            "a not-nullable function pointer must not be laid out like a "
            "nullable raw pointer",
        )

        self.expect(
            "frame variable fnHolder",
            substrs=["FnHolder", "a = ", "b = ", "c = "],
        )
