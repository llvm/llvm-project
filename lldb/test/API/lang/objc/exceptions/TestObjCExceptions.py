# encoding: utf-8
"""
Test lldb Obj-C exception support.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCExceptionsTestCase(TestBase):

    @skipIf(compiler="clang", compiler_version=['<', '13.0'])
    def test_objc_exceptions_at_throw(self):
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        launch_info = lldb.SBLaunchInfo(["a.out", "0"])
        launch_info.SetLaunchFlags(lldb.eLaunchFlagInheritTCCFromParent)
        lldbutil.run_to_name_breakpoint(self, "objc_exception_throw", launch_info=launch_info)

        self.expect("thread list",
            substrs=['stopped', 'stop reason = hit Objective-C exception'])

        self.expect('thread exception', substrs=[
                '(NSException *) exception = ',
                '"SomeReason"',
            ])

        target = self.dbg.GetSelectedTarget()
        thread = target.GetProcess().GetSelectedThread()
        frame = thread.GetSelectedFrame()

        opts = lldb.SBVariablesOptions()
        opts.SetIncludeRecognizedArguments(True)
        variables = frame.GetVariables(opts)

        self.assertEqual(variables.GetSize(), 1)
        self.assertEqual(variables.GetValueAtIndex(0).name, "exception")
        self.assertEqual(variables.GetValueAtIndex(0).GetValueType(), lldb.eValueTypeVariableArgument)

        lldbutil.run_to_source_breakpoint(self, "// Set break point at this line.", lldb.SBFileSpec("main.mm"), launch_info=launch_info)

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        target = self.dbg.GetSelectedTarget()
        thread = target.GetProcess().GetSelectedThread()
        frame = thread.GetSelectedFrame()

        # No exception being currently thrown/caught at this point
        self.assertFalse(thread.GetCurrentException().IsValid())
        self.assertFalse(thread.GetCurrentExceptionBacktrace().IsValid())

        self.expect(
            'frame variable e1',
            substrs=[
                '(NSException *) e1 = ',
                '"SomeReason"'
            ])

        self.expect(
            'frame variable *e1',
            substrs=[
                '(NSException) *e1 = ',
                'name = ', '"ExceptionName"',
                'reason = ', '"SomeReason"',
                'userInfo = ', '1 key/value pair',
                'reserved = ', 'nil',
            ])

        e1 = frame.FindVariable("e1")
        self.assertTrue(e1)
        self.assertEqual(e1.type.name, "NSException *")
        self.assertEqual(e1.GetSummary(), '"SomeReason"')
        self.assertEqual(e1.GetChildMemberWithName("name").description, "ExceptionName")
        self.assertEqual(e1.GetChildMemberWithName("reason").description, "SomeReason")
        userInfo = e1.GetChildMemberWithName("userInfo").dynamic
        self.assertEqual(userInfo.summary, "1 key/value pair")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(0).description, "some_key")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(1).description, "some_value")
        self.assertEqual(e1.GetChildMemberWithName("reserved").description, "<nil>")

        self.expect(
            'frame variable e2',
            substrs=[
                '(NSException *) e2 = ',
                '"SomeReason"'
            ])

        self.expect(
            'frame variable *e2',
            substrs=[
                '(NSException) *e2 = ',
                'name = ', '"ThrownException"',
                'reason = ', '"SomeReason"',
                'userInfo = ', '1 key/value pair',
                'reserved = ',
            ])

        e2 = frame.FindVariable("e2")
        self.assertTrue(e2)
        self.assertEqual(e2.type.name, "NSException *")
        self.assertEqual(e2.GetSummary(), '"SomeReason"')
        self.assertEqual(e2.GetChildMemberWithName("name").description, "ThrownException")
        self.assertEqual(e2.GetChildMemberWithName("reason").description, "SomeReason")
        userInfo = e2.GetChildMemberWithName("userInfo").dynamic
        self.assertEqual(userInfo.summary, "1 key/value pair")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(0).description, "some_key")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(1).description, "some_value")
        reserved = e2.GetChildMemberWithName("reserved").dynamic
        self.assertGreater(reserved.num_children, 0)
        callStackReturnAddresses = [reserved.GetChildAtIndex(i).GetChildAtIndex(1) for i in range(0, reserved.GetNumChildren())
                if reserved.GetChildAtIndex(i).GetChildAtIndex(0).description == "callStackReturnAddresses"][0].dynamic
        children = [callStackReturnAddresses.GetChildAtIndex(i) for i in range(0, callStackReturnAddresses.num_children)]

        pcs = [i.unsigned for i in children]
        names = [target.ResolveSymbolContextForAddress(lldb.SBAddress(pc, target), lldb.eSymbolContextSymbol).GetSymbol().name for pc in pcs]
        for n in ["objc_exception_throw", "foo(int)", "main"]:
            self.assertIn(n, names, "%s is in the exception backtrace (%s)" % (n, names))

    @skipIf(compiler="clang", compiler_version=['<', '13.0'])
    def test_objc_exceptions_at_abort(self):
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.runCmd("run 0")

        # We should be stopped at pthread_kill because of an unhandled exception
        self.expect("thread list",
            substrs=['stopped', 'stop reason = signal SIGABRT'])

        self.expect('thread exception', substrs=[
                '(NSException *) exception = ',
                '"SomeReason"',
                'libobjc.A.dylib`objc_exception_throw',
                'a.out`foo', 'at main.mm:16',
                'a.out`rethrow', 'at main.mm:27',
                'a.out`main',
            ])

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()

        # There is an exception being currently processed at this point
        self.assertTrue(thread.GetCurrentException().IsValid())
        self.assertTrue(thread.GetCurrentExceptionBacktrace().IsValid())

        history_thread = thread.GetCurrentExceptionBacktrace()
        self.assertGreaterEqual(history_thread.num_frames, 4)
        for n in ["objc_exception_throw", "foo(int)", "rethrow(int)", "main"]:
            self.assertEqual(len([f for f in history_thread.frames if f.GetFunctionName() == n]), 1)

        self.runCmd("kill")

        self.runCmd("run 1")
        # We should be stopped at pthread_kill because of an unhandled exception
        self.expect("thread list",
            substrs=['stopped', 'stop reason = signal SIGABRT'])

        self.expect('thread exception', substrs=[
                '(MyCustomException *) exception = ',
                'libobjc.A.dylib`objc_exception_throw',
                'a.out`foo', 'at main.mm:18',
                'a.out`rethrow', 'at main.mm:27',
                'a.out`main',
            ])

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()

        history_thread = thread.GetCurrentExceptionBacktrace()
        self.assertGreaterEqual(history_thread.num_frames, 4)
        for n in ["objc_exception_throw", "foo(int)", "rethrow(int)", "main"]:
            self.assertEqual(len([f for f in history_thread.frames if f.GetFunctionName() == n]), 1)

    @skipIf(compiler="clang", compiler_version=['<', '13.0'])
    def test_cxx_exceptions_at_abort(self):
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.runCmd("run 2")

        # We should be stopped at pthread_kill because of an unhandled exception
        self.expect("thread list",
            substrs=['stopped', 'stop reason = signal SIGABRT'])

        self.expect('thread exception', substrs=['exception ='])

        process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()

        self.assertTrue(thread.GetCurrentException().IsValid())

        # C++ exception backtraces are not exposed in the API (yet).
        self.assertFalse(thread.GetCurrentExceptionBacktrace().IsValid())
