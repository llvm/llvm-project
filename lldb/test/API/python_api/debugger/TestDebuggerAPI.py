"""
Test Debugger APIs.
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DebuggerAPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_debugger_api_boundary_condition(self):
        """Exercise SBDebugger APIs with boundary conditions."""
        self.dbg.HandleCommand(None)
        self.dbg.SetDefaultArchitecture(None)
        self.dbg.GetScriptingLanguage(None)
        self.dbg.CreateTarget(None)
        self.dbg.CreateTarget(None, None, None, True, lldb.SBError())
        self.dbg.CreateTargetWithFileAndTargetTriple(None, None)
        self.dbg.CreateTargetWithFileAndArch(None, None)
        self.dbg.FindTargetWithFileAndArch(None, None)
        self.dbg.SetInternalVariable(None, None, None)
        self.dbg.GetInternalVariableValue(None, None)
        # FIXME (filcab): We must first allow for the swig bindings to know if
        # a Python callback is set. (Check python-typemaps.swig)
        # self.dbg.SetLoggingCallback(None)
        self.dbg.SetPrompt(None)
        self.dbg.SetCurrentPlatform(None)
        self.dbg.SetCurrentPlatformSDKRoot(None)

        fresh_dbg = lldb.SBDebugger()
        self.assertEqual(len(fresh_dbg), 0)

    def test_debugger_delete_invalid_target(self):
        """SBDebugger.DeleteTarget() should not crash LLDB given and invalid target."""
        target = lldb.SBTarget()
        self.assertFalse(target.IsValid())
        self.dbg.DeleteTarget(target)

    def test_debugger_internal_variables(self):
        """Ensure that SBDebugger reachs the same instance of properties
        regardless CommandInterpreter's context initialization"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        property_name = "target.process.memory-cache-line-size"

        def get_cache_line_size():
            value_list = lldb.SBStringList()
            value_list = self.dbg.GetInternalVariableValue(
                property_name, self.dbg.GetInstanceName()
            )

            self.assertEqual(value_list.GetSize(), 1)
            try:
                return int(value_list.GetStringAtIndex(0))
            except ValueError as error:
                self.fail("Value is not a number: " + error)

        # Get global property value while there are no processes.
        global_cache_line_size = get_cache_line_size()

        # Run a process via SB interface. CommandInterpreter's execution context
        # remains empty.
        error = lldb.SBError()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetLaunchFlags(lldb.eLaunchFlagStopAtEntry)
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # This should change the value of a process's local property.
        new_cache_line_size = global_cache_line_size + 512
        error = self.dbg.SetInternalVariable(
            property_name, str(new_cache_line_size), self.dbg.GetInstanceName()
        )
        self.assertSuccess(error, property_name + " value was changed successfully")

        # Check that it was set actually.
        self.assertEqual(get_cache_line_size(), new_cache_line_size)

        # Run any command to initialize CommandInterpreter's execution context.
        self.runCmd("target list")

        # Test the local property again, is it set to new_cache_line_size?
        self.assertEqual(get_cache_line_size(), new_cache_line_size)

    @expectedFailureAll(
        remote=True,
        bugnumber="github.com/llvm/llvm-project/issues/92419",
    )
    def test_CreateTarget_platform(self):
        exe = self.getBuildArtifact("a.out")
        self.yaml2obj("elf.yaml", exe)
        error = lldb.SBError()
        target1 = self.dbg.CreateTarget(exe, None, "remote-linux", False, error)
        self.assertSuccess(error)
        platform1 = target1.GetPlatform()
        platform1.SetWorkingDirectory("/foo/bar")

        # Reuse a platform if it matches the currently selected one...
        target2 = self.dbg.CreateTarget(exe, None, "remote-linux", False, error)
        self.assertSuccess(error)
        platform2 = target2.GetPlatform()
        self.assertTrue(
            platform2.GetWorkingDirectory().endswith("bar"),
            platform2.GetWorkingDirectory(),
        )

        # ... but create a new one if it doesn't.
        self.dbg.SetSelectedPlatform(lldb.SBPlatform("remote-windows"))
        target3 = self.dbg.CreateTarget(exe, None, "remote-linux", False, error)
        self.assertSuccess(error)
        platform3 = target3.GetPlatform()
        self.assertIsNone(platform3.GetWorkingDirectory())

    def test_CreateTarget_arch(self):
        exe = self.getBuildArtifact("a.out")
        if lldbplatformutil.getHostPlatform() == "linux":
            self.yaml2obj("macho.yaml", exe)
            arch = "x86_64-apple-macosx"
            expected_platform = "remote-macosx"
        else:
            self.yaml2obj("elf.yaml", exe)
            arch = "x86_64-pc-linux"
            expected_platform = "remote-linux"

        fbsd = lldb.SBPlatform("remote-freebsd")
        self.dbg.SetSelectedPlatform(fbsd)

        error = lldb.SBError()
        target1 = self.dbg.CreateTarget(exe, arch, None, False, error)
        self.assertSuccess(error)
        platform1 = target1.GetPlatform()
        self.assertEqual(platform1.GetName(), expected_platform)
        platform1.SetWorkingDirectory("/foo/bar")

        # Reuse a platform even if it is not currently selected.
        self.dbg.SetSelectedPlatform(fbsd)
        target2 = self.dbg.CreateTarget(exe, arch, None, False, error)
        self.assertSuccess(error)
        platform2 = target2.GetPlatform()
        self.assertEqual(platform2.GetName(), expected_platform)
        self.assertTrue(
            platform2.GetWorkingDirectory().endswith("bar"),
            platform2.GetWorkingDirectory(),
        )

    def test_SetDestroyCallback(self):
        destroy_dbg_id = None

        def foo(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal destroy_dbg_id
            destroy_dbg_id = dbg_id

        self.dbg.SetDestroyCallback(foo)

        original_dbg_id = self.dbg.GetID()
        self.dbg.Destroy(self.dbg)
        self.assertEqual(destroy_dbg_id, original_dbg_id)

    def test_AddDestroyCallback(self):
        original_dbg_id = self.dbg.GetID()
        called = []

        def foo(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal called
            called += [('foo', dbg_id)]

        def bar(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal called
            called += [('bar', dbg_id)]

        token_foo = self.dbg.AddDestroyCallback(foo)
        token_bar = self.dbg.AddDestroyCallback(bar)
        self.dbg.Destroy(self.dbg)

        # Should call both `foo()` and `bar()`.
        self.assertEqual(called, [
            ('foo', original_dbg_id),
            ('bar', original_dbg_id),
        ])

    def test_RemoveDestroyCallback(self):
        original_dbg_id = self.dbg.GetID()
        called = []

        def foo(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal called
            called += [('foo', dbg_id)]

        def bar(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal called
            called += [('bar', dbg_id)]

        token_foo = self.dbg.AddDestroyCallback(foo)
        token_bar = self.dbg.AddDestroyCallback(bar)
        ret = self.dbg.RemoveDestroyCallback(token_foo)
        self.dbg.Destroy(self.dbg)

        # `Remove` should be successful
        self.assertTrue(ret)
        # Should only call `bar()`
        self.assertEqual(called, [('bar', original_dbg_id)])

    def test_RemoveDestroyCallback_invalid_token(self):
        original_dbg_id = self.dbg.GetID()
        magic_token_that_should_not_exist = 32413
        called = []

        def foo(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal called
            called += [('foo', dbg_id)]

        token_foo = self.dbg.AddDestroyCallback(foo)
        ret = self.dbg.RemoveDestroyCallback(magic_token_that_should_not_exist)
        self.dbg.Destroy(self.dbg)

        # `Remove` should be unsuccessful
        self.assertFalse(ret)
        # Should call `foo()`
        self.assertEqual(called, [('foo', original_dbg_id)])

    def test_HandleDestroyCallback(self):
        """
        Validates:
        1. AddDestroyCallback and RemoveDestroyCallback work during debugger destroy.
        2. HandleDestroyCallback invokes all callbacks in FIFO order.
        """
        original_dbg_id = self.dbg.GetID()
        events = []
        bar_token = None

        def foo(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal events
            events.append(('foo called', dbg_id))

        def bar(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal events
            events.append(('bar called', dbg_id))

        def add_foo(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal events
            events.append(('add_foo called', dbg_id))
            events.append(('foo token', self.dbg.AddDestroyCallback(foo)))

        def remove_bar(dbg_id):
            # Need nonlocal to modify closure variable.
            nonlocal events
            events.append(('remove_bar called', dbg_id))
            events.append(('remove bar ret', self.dbg.RemoveDestroyCallback(bar_token)))

        # Setup
        events.append(('add_foo token', self.dbg.AddDestroyCallback(add_foo)))
        bar_token = self.dbg.AddDestroyCallback(bar)
        events.append(('bar token', bar_token))
        events.append(('remove_bar token', self.dbg.AddDestroyCallback(remove_bar)))
        # Destroy
        self.dbg.Destroy(self.dbg)

        self.assertEqual(events, [
            # Setup
            ('add_foo token', 0), # add_foo should be added
            ('bar token', 1), # bar should be added
            ('remove_bar token', 2), # remove_bar should be added
            # Destroy
            ('add_foo called', original_dbg_id), # add_foo should be called
            ('foo token', 3), # foo should be added
            ('bar called', original_dbg_id), # bar should be called
            ('remove_bar called', original_dbg_id), # remove_bar should be called
            ('remove bar ret', False), # remove_bar should fail, because it's already invoked and removed
            ('foo called', original_dbg_id), # foo should be called
        ])
