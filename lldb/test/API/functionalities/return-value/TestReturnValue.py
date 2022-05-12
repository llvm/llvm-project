"""
Test getting return-values correctly when stepping out
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ReturnValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def affected_by_pr33042(self):
        return ("clang" in self.getCompiler() and self.isAArch64() and
            self.getPlatform() == "linux")

    def affected_by_pr44132(self):
        return (self.getArchitecture() in ["aarch64", "arm"] and
                self.getPlatform() in ["freebsd", "linux"])

    # ABIMacOSX_arm64 and the SysV_arm64 don't restore the storage value for memory returns on function
    # exit, so lldb shouldn't attempt to fetch memory for those return types, as there is
    # no easy way to guarantee that they will be correct.  This is a list of the memory
    # return functions defined in the test file:
    arm_no_return_values = ["return_five_int", "return_one_int_one_double_one_int",
                            "return_one_short_one_double_one_short", "return_vector_size_float32_32",
                            "return_ext_vector_size_float32_8"]
    def should_report_return_value(self, func_name):
        abi = self.target.GetABIName()
        if not abi in ["SysV-arm64", "ABIMacOSX_arm64", "macosx-arm"]:
            return True
        return not func_name in self.arm_no_return_values

    @expectedFailureAll(oslist=["freebsd"], archs=["i386"],
                        bugnumber="llvm.org/pr48376")
    @expectedFailureAll(oslist=["macosx"], archs=["i386"], bugnumber="<rdar://problem/28719652>")
    @expectedFailureAll(
        oslist=["linux"],
        compiler="clang",
        compiler_version=[
            "<=",
            "3.6"],
        archs=["i386"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    @add_test_categories(['pyapi'])
    def test_with_python(self):
        """Test getting return values from stepping out."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        (self.target, self.process, thread, inner_sint_bkpt) = lldbutil.run_to_name_breakpoint(self, "inner_sint", exe_name = exe)

        error = lldb.SBError()

        # inner_sint returns the variable value, so capture that here:
        in_int = thread.GetFrameAtIndex(0).FindVariable(
            "value").GetValueAsSigned(error)
        self.assertSuccess(error)

        thread.StepOut()

        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertEquals(fun_name, "outer_sint(int)")

        return_value = thread.GetStopReturnValue()
        self.assertTrue(return_value.IsValid())

        ret_int = return_value.GetValueAsSigned(error)
        self.assertSuccess(error)
        self.assertEquals(in_int, ret_int)

        # Run again and we will stop in inner_sint the second time outer_sint is called.
        # Then test stepping out two frames at once:

        thread_list = lldbutil.continue_to_breakpoint(self.process, inner_sint_bkpt)
        self.assertEquals(len(thread_list), 1)
        thread = thread_list[0]

        # We are done with the inner_sint breakpoint:
        self.target.BreakpointDelete(inner_sint_bkpt.GetID())

        frame = thread.GetFrameAtIndex(1)
        fun_name = frame.GetFunctionName()
        self.assertEquals(fun_name, "outer_sint(int)")
        in_int = frame.FindVariable("value").GetValueAsSigned(error)
        self.assertSuccess(error)

        thread.StepOutOfFrame(frame)

        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(thread.GetStopReason(), lldb.eStopReasonPlanComplete)
        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertEquals(fun_name, "main")

        ret_value = thread.GetStopReturnValue()
        self.assertTrue(return_value.IsValid())
        ret_int = ret_value.GetValueAsSigned(error)
        self.assertSuccess(error)
        self.assertEquals(2 * in_int, ret_int)

        # Now try some simple returns that have different types:
        inner_float_bkpt = self.target.BreakpointCreateByName(
            "inner_float(float)", exe)
        self.assertTrue(inner_float_bkpt, VALID_BREAKPOINT)
        self.process.Continue()
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, inner_float_bkpt)
        self.assertEquals(len(thread_list), 1)
        thread = thread_list[0]

        self.target.BreakpointDelete(inner_float_bkpt.GetID())

        frame = thread.GetFrameAtIndex(0)
        in_value = frame.FindVariable("value")
        in_float = float(in_value.GetValue())
        thread.StepOut()

        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertEquals(fun_name, "outer_float(float)")

        #return_value = thread.GetStopReturnValue()
        #self.assertTrue(return_value.IsValid())
        #return_float = float(return_value.GetValue())

        #self.assertEqual(in_float, return_float)

        if not self.affected_by_pr44132():
            self.return_and_test_struct_value("return_one_int")
            self.return_and_test_struct_value("return_two_int")
            self.return_and_test_struct_value("return_three_int")
            self.return_and_test_struct_value("return_four_int")
            if not self.affected_by_pr33042():
                self.return_and_test_struct_value("return_five_int")
            self.return_and_test_struct_value("return_two_double")
            self.return_and_test_struct_value("return_one_double_two_float")
            self.return_and_test_struct_value("return_one_int_one_float_one_int")

            self.return_and_test_struct_value("return_one_pointer")
            self.return_and_test_struct_value("return_two_pointer")
            self.return_and_test_struct_value("return_one_float_one_pointer")
            self.return_and_test_struct_value("return_one_int_one_pointer")
            self.return_and_test_struct_value("return_three_short_one_float")

            self.return_and_test_struct_value("return_one_int_one_double")
            self.return_and_test_struct_value("return_one_int_one_double_one_int")
            self.return_and_test_struct_value(
                "return_one_short_one_double_one_short")
            self.return_and_test_struct_value("return_one_float_one_int_one_float")
            self.return_and_test_struct_value("return_two_float")
            # I am leaving out the packed test until we have a way to tell CLANG
            # about alignment when reading DWARF for packed types.
            #self.return_and_test_struct_value ("return_one_int_one_double_packed")
            self.return_and_test_struct_value("return_one_int_one_long")

    @expectedFailureAll(oslist=["freebsd"], archs=["i386"],
                        bugnumber="llvm.org/pr48376")
    @expectedFailureAll(oslist=["macosx"], archs=["i386"], bugnumber="<rdar://problem/28719652>")
    @expectedFailureAll(
        oslist=["linux"],
        compiler="clang",
        compiler_version=[
            "<=",
            "3.6"],
        archs=["i386"])
    @expectedFailureAll(compiler=["gcc"], archs=["x86_64", "i386"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_vector_values(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        error = lldb.SBError()

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        main_bktp = self.target.BreakpointCreateByName("main", exe)
        self.assertTrue(main_bktp, VALID_BREAKPOINT)

        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertEqual(len(lldbutil.get_threads_stopped_at_breakpoint(
            self.process, main_bktp)), 1)
        self.return_and_test_struct_value("return_vector_size_float32_8")
        self.return_and_test_struct_value("return_vector_size_float32_16")
        if not self.affected_by_pr44132():
            self.return_and_test_struct_value("return_vector_size_float32_32")
        self.return_and_test_struct_value("return_ext_vector_size_float32_2")
        self.return_and_test_struct_value("return_ext_vector_size_float32_4")
        if not self.affected_by_pr44132():
            self.return_and_test_struct_value("return_ext_vector_size_float32_8")

    # limit the nested struct and class tests to only x86_64
    @skipIf(archs=no_match(['x86_64']))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_for_cpp_support(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        (self.target, self.process, thread, inner_sint_bkpt) = lldbutil.run_to_name_breakpoint(self, "inner_sint", exe_name = exe)

        error = lldb.SBError()

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        main_bktp = self.target.BreakpointCreateByName("main", exe)
        self.assertTrue(main_bktp, VALID_BREAKPOINT)

        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertEqual(len(lldbutil.get_threads_stopped_at_breakpoint(
            self.process, main_bktp)), 1)
        # nested struct tests
        self.return_and_test_struct_value("return_nested_one_float_three_base")
        self.return_and_test_struct_value("return_double_nested_one_float_one_nested")
        self.return_and_test_struct_value("return_nested_float_struct")
        # class test
        self.return_and_test_struct_value("return_base_class_one_char")
        self.return_and_test_struct_value("return_nested_class_float_and_base")
        self.return_and_test_struct_value("return_double_nested_class_float_and_nested")
        self.return_and_test_struct_value("return_base_class")
        self.return_and_test_struct_value("return_derived_class")

    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    def return_and_test_struct_value(self, func_name):
        """Pass in the name of the function to return from - takes in value, returns value."""

        # Set the breakpoint, run to it, finish out.
        bkpt = self.target.BreakpointCreateByName(func_name)
        self.assertTrue(bkpt.GetNumResolvedLocations() > 0, "Got wrong number of locations for {0}".format(func_name))

        self.process.Continue()

        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, bkpt)

        self.assertEquals(len(thread_list), 1)
        thread = thread_list[0]

        self.target.BreakpointDelete(bkpt.GetID())

        in_value = thread.GetFrameAtIndex(0).FindVariable("value")

        self.assertTrue(in_value.IsValid())
        num_in_children = in_value.GetNumChildren()

        # This is a little hokey, but if we don't get all the children now, then
        # once we've stepped we won't be able to get them?

        for idx in range(0, num_in_children):
            in_child = in_value.GetChildAtIndex(idx)
            in_child_str = in_child.GetValue()

        thread.StepOut()

        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # Assuming all these functions step out to main.  Could figure out the caller dynamically
        # if that would add something to the test.
        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertEquals(fun_name, "main")

        frame = thread.GetFrameAtIndex(0)
        ret_value = thread.GetStopReturnValue()
        if not self.should_report_return_value(func_name):
            self.assertFalse(ret_value.IsValid(), "Shouldn't have gotten a value")
            return

        self.assertTrue(ret_value.IsValid())

        num_ret_children = ret_value.GetNumChildren()
        self.assertEquals(num_in_children, num_ret_children)
        for idx in range(0, num_ret_children):
            in_child = in_value.GetChildAtIndex(idx)
            ret_child = ret_value.GetChildAtIndex(idx)
            in_child_str = in_child.GetValue()
            ret_child_str = ret_child.GetValue()

            self.assertEqual(in_child_str, ret_child_str)
