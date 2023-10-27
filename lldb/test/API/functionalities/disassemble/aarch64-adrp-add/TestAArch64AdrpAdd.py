"""
Test that the arm64 ADRP + ADD pc-relative addressing pair is symbolicated.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestAArch64AdrpAdd(TestBase):
    @no_debug_info_test
    @skipIfLLVMTargetMissing("AArch64")
    def test_arm64(self):
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "a.out-arm64.yaml")
        obj_path = self.getBuildArtifact("a.out-arm64")
        self.yaml2obj(yaml_path, obj_path)

        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        mains = target.FindFunctions("main")
        for f in mains.symbols:
            binaryname = f.GetStartAddress().GetModule().GetFileSpec().GetFilename()
            if binaryname == "a.out-arm64":
                self.disassemble_check_for_hi_and_foo(target, f, binaryname)

    @no_debug_info_test
    @skipIfLLVMTargetMissing("AArch64")
    def test_arm64_32(self):
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "a.out-arm64_32.yaml")
        obj_path = self.getBuildArtifact("a.out-arm64_32")
        self.yaml2obj(yaml_path, obj_path)

        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        mains = target.FindFunctions("main")
        for f in mains.symbols:
            binaryname = f.GetStartAddress().GetModule().GetFileSpec().GetFilename()
            if binaryname == "a.out-arm64_32":
                self.disassemble_check_for_hi_and_foo(target, f, binaryname)

    def disassemble_check_for_hi_and_foo(self, target, func, binaryname):
        insns = func.GetInstructions(target)
        found_hi_string = False
        found_foo = False

        # The binary has an ADRP + ADD instruction pair which load
        # the pc-relative address of a c-string, and loads the address
        # of a function into a function pointer.  lldb should show
        # that c-string and the name of that function in the disassembly
        # comment field.
        for i in insns:
            if "HI" in i.GetComment(target):
                found_hi_string = True
            if "foo" in i.GetComment(target):
                found_foo = True
        if found_hi_string == False or found_foo == False:
            print(
                'Did not find "HI" string or "foo" in disassembly symbolication in %s'
                % binaryname
            )
            if self.TraceOn():
                strm = lldb.SBStream()
                insns.GetDescription(strm)
                print('Disassembly of main(), looking for "HI" and "foo" in comments:')
                print(strm.GetData())
        self.assertTrue(found_hi_string)
        self.assertTrue(found_foo)
