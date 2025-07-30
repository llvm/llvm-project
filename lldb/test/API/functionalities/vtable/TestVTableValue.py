"""
Make sure the getting a variable path works and doesn't crash.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestVTableValue(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(compiler="clang", compiler_version=["<", "9.0"])
    @skipUnlessPlatform(["linux", "macosx"])
    def test_vtable(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "At the end", lldb.SBFileSpec("main.cpp")
        )

        # Test a shape instance to make sure we get the vtable correctly.
        shape = self.frame().FindVariable("shape")
        vtable = shape.GetVTable()
        self.assertEqual(vtable.GetName(), "vtable for Shape")
        self.assertEqual(vtable.GetTypeName(), "vtable for Shape")
        # Make sure we have the right number of virtual functions in our vtable
        # for the shape class.
        self.assertEqual(vtable.GetNumChildren(), 4)

        # Verify vtable address
        vtable_addr = vtable.GetValueAsUnsigned(0)
        expected_addr = self.expected_vtable_addr(shape)
        self.assertEqual(vtable_addr, expected_addr)

        for idx, vtable_entry in enumerate(vtable.children):
            self.verify_vtable_entry(vtable_entry, vtable_addr, idx)

        # Test a shape reference to make sure we get the vtable correctly.
        shape = self.frame().FindVariable("shape_ref")
        vtable = shape.GetVTable()
        self.assertEqual(vtable.GetName(), "vtable for Shape")
        self.assertEqual(vtable.GetTypeName(), "vtable for Shape")
        # Make sure we have the right number of virtual functions in our vtable
        # for the shape class.
        self.assertEqual(vtable.GetNumChildren(), 4)

        # Verify vtable address
        vtable_addr = vtable.GetValueAsUnsigned(0)
        expected_addr = self.expected_vtable_addr(shape)
        self.assertEqual(vtable_addr, expected_addr)

        for idx, vtable_entry in enumerate(vtable.children):
            self.verify_vtable_entry(vtable_entry, vtable_addr, idx)

        # Test we get the right vtable for the Rectangle instance.
        rect = self.frame().FindVariable("rect")
        vtable = rect.GetVTable()
        self.assertEqual(vtable.GetName(), "vtable for Rectangle")
        self.assertEqual(vtable.GetTypeName(), "vtable for Rectangle")

        # Make sure we have the right number of virtual functions in our vtable
        # with the extra virtual function added by the Rectangle class
        self.assertEqual(vtable.GetNumChildren(), 5)

        # Verify vtable address
        vtable_addr = vtable.GetValueAsUnsigned()
        expected_addr = self.expected_vtable_addr(rect)
        self.assertEqual(vtable_addr, expected_addr)

        for idx, vtable_entry in enumerate(vtable.children):
            self.verify_vtable_entry(vtable_entry, vtable_addr, idx)

    @skipIf(compiler="clang", compiler_version=["<", "9.0"])
    @skipUnlessPlatform(["linux", "macosx"])
    def test_base_class_ptr(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Shape is Rectangle", lldb.SBFileSpec("main.cpp")
        )

        shape = self.frame().FindVariable("shape")
        rect = self.frame().FindVariable("rect")

        shape_ptr = self.frame().FindVariable("shape_ptr")
        shape_ptr_vtable = shape_ptr.GetVTable()
        self.assertEqual(shape_ptr_vtable.GetName(), "vtable for Rectangle")
        self.assertEqual(shape_ptr_vtable.GetNumChildren(), 5)
        self.assertEqual(shape_ptr.GetValueAsUnsigned(0), rect.GetLoadAddress())
        lldbutil.continue_to_source_breakpoint(
            self, process, "Shape is Shape", lldb.SBFileSpec("main.cpp")
        )
        self.assertEqual(shape_ptr.GetValueAsUnsigned(0), shape.GetLoadAddress())
        self.assertEqual(shape_ptr_vtable.GetNumChildren(), 4)
        self.assertEqual(shape_ptr_vtable.GetName(), "vtable for Shape")

    @skipUnlessPlatform(["linux", "macosx"])
    def test_no_vtable(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "At the end", lldb.SBFileSpec("main.cpp")
        )

        var = self.frame().FindVariable("not_virtual")
        self.assertEqual(
            var.GetVTable().GetError().GetCString(),
            'type "NotVirtual" doesn\'t have a vtable',
        )

        var = self.frame().FindVariable("argc")
        self.assertEqual(
            var.GetVTable().GetError().GetCString(),
            'no language runtime support for the language "c"',
        )

    @skipUnlessPlatform(["linux", "macosx"])
    def test_overwrite_vtable(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "At the end", lldb.SBFileSpec("main.cpp")
        )

        # Test a shape instance to make sure we get the vtable correctly.
        shape = self.frame().FindVariable("shape")
        vtable = shape.GetVTable()
        self.assertEqual(vtable.GetName(), "vtable for Shape")
        self.assertEqual(vtable.GetTypeName(), "vtable for Shape")
        # Make sure we have the right number of virtual functions in our vtable
        # for the shape class.
        self.assertEqual(vtable.GetNumChildren(), 4)

        # Overwrite the first entry in the vtable and make sure we can still
        # see the bogus value which should have no summary
        vtable_addr = vtable.GetValueAsUnsigned()

        is_64bit = self.process().GetAddressByteSize() == 8
        data = str(
            "\x01\x01\x01\x01\x01\x01\x01\x01" if is_64bit else "\x01\x01\x01\x01"
        )
        error = lldb.SBError()
        bytes_written = process.WriteMemory(vtable_addr, data, error)

        self.assertSuccess(error)
        self.assertGreater(
            bytes_written, 0, "Failed to overwrite first entry in vtable"
        )

        scribbled_child = vtable.GetChildAtIndex(0)
        self.assertEqual(
            scribbled_child.GetValueAsUnsigned(0),
            0x0101010101010101 if is_64bit else 0x01010101,
        )
        self.assertEqual(scribbled_child.GetSummary(), None)

    def expected_vtable_addr(self, var: lldb.SBValue) -> int:
        load_addr = var.GetLoadAddress()
        read_from_memory_error = lldb.SBError()
        vtable_addr = self.process().ReadPointerFromMemory(
            load_addr, read_from_memory_error
        )
        self.assertTrue(read_from_memory_error.Success())
        return vtable_addr

    def expected_vtable_entry_func_ptr(self, vtable_addr: int, idx: int):
        vtable_entry_addr = vtable_addr + idx * self.process().GetAddressByteSize()
        read_func_ptr_error = lldb.SBError()
        func_ptr = self.process().ReadPointerFromMemory(
            vtable_entry_addr, read_func_ptr_error
        )
        self.assertTrue(read_func_ptr_error.Success())
        return func_ptr

    def verify_vtable_entry(
        self, vtable_entry: lldb.SBValue, vtable_addr: int, idx: int
    ):
        """Verify the vtable entry looks something like:

        (double ()) [0] = 0x0000000100003a10 a.out`Rectangle::Area() at main.cpp:14

        """
        # Check function ptr
        vtable_entry_func_ptr = vtable_entry.GetValueAsUnsigned(0)
        self.assertEqual(
            vtable_entry_func_ptr,
            self.expected_vtable_entry_func_ptr(vtable_addr, idx),
        )

        sb_addr = self.target().ResolveLoadAddress(vtable_entry_func_ptr)
        sym_ctx = sb_addr.GetSymbolContext(lldb.eSymbolContextEverything)

        # Make sure the type is the same as the function type
        func_type = sym_ctx.GetFunction().GetType()
        if func_type.IsValid():
            self.assertEqual(vtable_entry.GetType(), func_type.GetPointerType())

        # The summary should be the address description of the function pointer
        summary = vtable_entry.GetSummary()
        self.assertEqual(str(sb_addr), summary)
