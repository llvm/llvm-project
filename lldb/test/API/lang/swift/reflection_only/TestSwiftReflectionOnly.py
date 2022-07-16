import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2
import re

class TestSwiftReflectionOnly(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test debugging a program without swiftmodules is functional"""
        self.build()

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'),
            extra_images=['dynamic_lib'])
        log = self.getBuildArtifact('types.log')
        self.expect('log enable lldb types -f ' + log)

        check_var = lldbutil.check_variable
        frame = thread.frames[0]
        var_self = frame.FindVariable("self")
        var_self_x = var_self.GetChildMemberWithName("x")
        check_var(self, var_self_x, value="42")

        check_var(self, frame.FindVariable("number"), value="1")

        array = frame.FindVariable("array")
        check_var(self, array, num_children=3)
        check_var(self, array.GetChildAtIndex(0), value="1")
        check_var(self, array.GetChildAtIndex(1), value="2")
        check_var(self, array.GetChildAtIndex(2), value="3")

        check_var(self, frame.FindVariable("string"), summary='"hello"')

        tup = frame.FindVariable("tuple")
        check_var(self, tup, num_children=2)
        check_var(self, tup.GetChildAtIndex(0), value="0")
        check_var(self, tup.GetChildAtIndex(1), value="1")

        strct = frame.FindVariable("strct")
        check_var(self, strct, num_children=5)
        check_var(self, strct.GetChildMemberWithName("pub"), value="1")
        check_var(self, strct.GetChildMemberWithName("priv"), value="2")
        check_var(self, strct.GetChildMemberWithName("filepriv"), value="3")
        s_priv = strct.GetChildMemberWithName("s_priv")
        check_var(self, s_priv, num_children=1)
        check_var(self, s_priv.GetChildMemberWithName("i"), value="2")
        s_filepriv = strct.GetChildMemberWithName("s_filepriv")
        check_var(self, s_filepriv, num_children=1)
        check_var(self, s_filepriv.GetChildMemberWithName("i"), value="3")

        check_var(self, frame.FindVariable("generic"), use_dynamic=True, value="42")

        gtup = frame.FindVariable("generic_tuple")
        check_var(self, gtup, num_children=2)
        check_var(self, gtup.GetChildAtIndex(0), use_dynamic=True, value="42")
        check_var(self, gtup.GetChildAtIndex(1), use_dynamic=True, value="42")

        check_var(self, frame.FindVariable("word"), value="0")
        check_var(self, frame.FindVariable("enum1"), value="second")
        enum2 = frame.FindVariable("enum2")
        check_var(self, enum2, value="with")
        check_var(self, enum2, num_children=1)
        # FIXME:  Fails in swift::reflection::NoPayloadEnumTypeInfo::projectEnumValue: .second
        # check_var(self, enum2.GetChildAtIndex(0), value="42")

        # Scan through the types log.
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        found_ref_exe = 0
        found_ref_lib = 0
        found_ast_exe = 0
        found_ast_lib = 0
        for line in logfile:
            if 'SwiftASTContextForExpressions::RegisterSectionModules("a.out");' in line:
                if not 'retrieved 0 AST Data blobs' in line:
                    found_ast_exe += 1
            elif 'SwiftASTContextForExpressions::RegisterSectionModules("dyld")' in line:
                if not 'retrieved 0 AST Data blobs' in line:
                    found_ast_lib += 1
            elif re.search(r'Adding reflection metadata in .*a\.out', line):
                found_ref_exe += 1
            elif re.search(r'Adding reflection metadata in .*dynamic_lib', line):
                found_ref_lib += 1
        self.assertEqual(found_ref_exe, 1)
        self.assertEqual(found_ref_lib, 1)
        self.assertEqual(found_ast_exe, 0)
        self.assertEqual(found_ast_lib, 0)
