"""
Test the "target modules replace" command.
"""

import os
import shutil

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@skipIfWindows
class TargetModulesReplaceTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        # The "v2" variant of the library is built into a subdirectory so that
        # it can share a soname with the "v1" variant.
        lldbutil.mkdir_p(self.getBuildArtifact("hidden"))

    def build_and_get_paths(self):
        """Build and return the paths of the two library variants, plus a copy
        of v1 at a third path. The copy shares v1's UUID, the v2 variant does
        not, which is what lets the two lookup paths be tested apart."""
        self.build()
        lib_name = self.platformContext.getFullLibName("replace_v")
        v1 = self.getBuildArtifact(lib_name)
        v2 = os.path.join(self.getBuildDir(), "hidden", lib_name)
        v1_copy = self.getBuildArtifact("copy_of_" + lib_name)
        shutil.copyfile(v1, v1_copy)
        for path in (v1, v2, v1_copy):
            self.assertTrue(os.path.exists(path), "%s was built" % path)
        return v1, v2, v1_copy

    def static_target_with_v1(self):
        """Make a target with the v1 library added but nothing loaded."""
        v1, v2, v1_copy = self.build_and_get_paths()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        self.runCmd("target modules add '%s'" % v1)
        return target, v1, v2, v1_copy

    def base_load_address(self, module, target):
        return module.GetObjectFileHeaderAddress().GetLoadAddress(target)

    def test_matching_uuid_finds_the_module(self):
        """A single argument is enough when the UUID identifies the module."""
        target, v1, v2, v1_copy = self.static_target_with_v1()
        num_modules = target.GetNumModules()

        # No --old-path: the copy shares v1's UUID, so the module to replace is
        # worked out from the file alone. --force because both are real modules
        # rather than placeholders.
        self.runCmd(
            "target modules replace --allow-uuid-mismatch --force '%s'" % v1_copy
        )

        self.assertEqual(target.GetNumModules(), num_modules)
        self.assertFalse(target.FindModule(lldb.SBFileSpec(v1)).IsValid())
        self.assertTrue(target.FindModule(lldb.SBFileSpec(v1_copy)).IsValid())

    def test_mismatched_uuid_is_an_error(self):
        """A file that is not the same build is refused unless forced."""
        target, v1, v2, v1_copy = self.static_target_with_v1()
        num_modules = target.GetNumModules()

        self.expect(
            "target modules replace '%s'" % v2,
            error=True,
            substrs=["does not match UUID", "--allow-uuid-mismatch"],
        )

        # The target must be left exactly as it was.
        self.assertEqual(target.GetNumModules(), num_modules)
        self.assertTrue(target.FindModule(lldb.SBFileSpec(v1)).IsValid())
        self.assertFalse(target.FindModule(lldb.SBFileSpec(v2)).IsValid())

        # And --force goes through.
        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)
        self.assertTrue(target.FindModule(lldb.SBFileSpec(v2)).IsValid())

    def test_old_path_option(self):
        """--old-path names the module to replace explicitly."""
        target, v1, v2, v1_copy = self.static_target_with_v1()

        self.runCmd(
            "target modules replace --old-path '%s' --allow-uuid-mismatch --force '%s'"
            % (v1, v2)
        )
        self.assertFalse(target.FindModule(lldb.SBFileSpec(v1)).IsValid())
        self.assertTrue(target.FindModule(lldb.SBFileSpec(v2)).IsValid())

    def test_real_module_needs_force(self):
        """Only unused placeholders are replaced by default."""
        target, v1, v2, v1_copy = self.static_target_with_v1()
        num_modules = target.GetNumModules()

        # v1_copy shares v1's UUID, so only the placeholder gate can refuse it.
        self.expect(
            "target modules replace '%s'" % v1_copy,
            error=True,
            substrs=["not an unused placeholder", "--force"],
        )
        self.assertEqual(target.GetNumModules(), num_modules)
        self.assertTrue(target.FindModule(lldb.SBFileSpec(v1)).IsValid())
        self.assertFalse(target.FindModule(lldb.SBFileSpec(v1_copy)).IsValid())

        self.runCmd("target modules replace --force '%s'" % v1_copy)
        self.assertTrue(target.FindModule(lldb.SBFileSpec(v1_copy)).IsValid())

    def test_no_matching_module(self):
        """A file that matches nothing points the user at --old-path."""
        target, v1, v2, v1_copy = self.static_target_with_v1()
        unrelated = self.getBuildArtifact("other.out")

        self.expect(
            "target modules replace '%s'" % unrelated,
            error=True,
            substrs=["no module in the target was found", "--old-path"],
        )

    def test_module_is_replaced_not_mutated(self):
        """The old module is removed and a distinct new one takes its place."""
        target, v1, v2, v1_copy = self.static_target_with_v1()

        old_module = target.FindModule(lldb.SBFileSpec(v1))
        self.assertTrue(old_module.IsValid(), "v1 is in the target")
        old_uuid = old_module.GetUUIDString()

        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)

        new_module = target.FindModule(lldb.SBFileSpec(v2))
        self.assertTrue(new_module.IsValid(), "v2 was added to the target")
        self.assertNotEqual(old_uuid, new_module.GetUUIDString())

        # Symbols now come from the replacement.
        self.assertTrue(new_module.FindSymbol("only_in_v2").IsValid())
        self.assertFalse(new_module.FindSymbol("only_in_v1").IsValid())

        # The old module object was not modified in place. This is the guard
        # against implementing the command by swapping the ObjectFile out from
        # under a live Module, which leaves stale pointers behind.
        self.assertEqual(old_module.GetUUIDString(), old_uuid)
        self.assertTrue(
            old_module.FindSymbol("only_in_v1").IsValid(),
            "the replaced module still describes its own file",
        )

    def test_unloaded_module_stays_unloaded(self):
        """Replacing a module that was never loaded doesn't load anything."""
        target, v1, v2, v1_copy = self.static_target_with_v1()

        old_module = target.FindModule(lldb.SBFileSpec(v1))
        self.assertEqual(
            self.base_load_address(old_module, target), lldb.LLDB_INVALID_ADDRESS
        )

        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)

        new_module = target.FindModule(lldb.SBFileSpec(v2))
        self.assertEqual(
            self.base_load_address(new_module, target), lldb.LLDB_INVALID_ADDRESS
        )
        for section in new_module.section_iter():
            self.assertEqual(section.GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)

    def test_load_address_is_preserved(self):
        """A loaded module's replacement is loaded at the same address."""
        target, v1, v2, v1_copy = self.static_target_with_v1()
        self.runCmd("target modules load --file '%s' --slide 0x100000" % v1)

        old_module = target.FindModule(lldb.SBFileSpec(v1))
        base_before = self.base_load_address(old_module, target)
        self.assertNotEqual(base_before, lldb.LLDB_INVALID_ADDRESS)

        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)

        # The image base is what is preserved. Individual section addresses are
        # not comparable: the two files lay their sections out differently.
        new_module = target.FindModule(lldb.SBFileSpec(v2))
        self.assertEqual(self.base_load_address(new_module, target), base_before)

        # The replaced module's sections were unloaded.
        for section in old_module.section_iter():
            self.assertEqual(
                section.GetLoadAddress(target),
                lldb.LLDB_INVALID_ADDRESS,
                "section %s of the replaced module was unloaded" % section.GetName(),
            )

    def test_replace_executable(self):
        """The replacement executable stays at the front of the module list."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        other = self.getBuildArtifact("other.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertEqual(
            target.GetModuleAtIndex(0).GetFileSpec().GetFilename(), "a.out"
        )
        num_modules = target.GetNumModules()

        self.runCmd(
            "target modules replace --old-path '%s' --allow-uuid-mismatch --force '%s'"
            % (exe, other)
        )

        self.assertEqual(target.GetNumModules(), num_modules)
        self.assertEqual(
            target.GetModuleAtIndex(0).GetFileSpec().GetFilename(),
            "other.out",
            "the replacement executable is at index 0",
        )
        self.assertEqual(
            target.GetExecutable().GetFilename(),
            "other.out",
            "the target's executable follows the replacement",
        )

    def test_round_trip(self):
        """Replacing back and forth doesn't accumulate or drop modules."""
        target, v1, v2, v1_copy = self.static_target_with_v1()
        num_modules = target.GetNumModules()

        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)
        self.assertEqual(target.GetNumModules(), num_modules)
        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v1)
        self.assertEqual(target.GetNumModules(), num_modules)

        self.assertTrue(target.FindModule(lldb.SBFileSpec(v1)).IsValid())
        self.assertFalse(target.FindModule(lldb.SBFileSpec(v2)).IsValid())

    def test_errors(self):
        """Bad input is rejected without touching the target."""
        target, v1, v2, v1_copy = self.static_target_with_v1()
        num_modules = target.GetNumModules()

        self.expect(
            "target modules replace /no/such/file",
            error=True,
            substrs=["invalid module path"],
        )
        self.expect(
            "target modules replace",
            error=True,
            substrs=["takes one argument"],
        )
        self.expect(
            "target modules replace --old-path /not/in/the/target '%s'" % v2,
            error=True,
            substrs=["no module in the target was found"],
        )

        self.assertEqual(target.GetNumModules(), num_modules)
        self.assertTrue(target.FindModule(lldb.SBFileSpec(v1)).IsValid())

    def test_placeholder_without_uuid(self):
        """A placeholder with no UUID needs no --force, nothing can be compared."""
        core = self.getBuildArtifact("no-uuid.dmp")
        self.yaml2obj("placeholder-no-uuid.yaml", core)
        # The dump is x86_64, so the replacement comes from a yaml too rather
        # than from this test's libraries, which follow the host architecture.
        replacement = self.getBuildArtifact("replacement.so")
        self.yaml2obj("replacement.yaml", replacement)

        target = self.dbg.CreateTarget("")
        self.assertTrue(target.LoadCore(core).IsValid())
        placeholder = target.FindModule(lldb.SBFileSpec("/no/such/module.so"))
        self.assertTrue(placeholder.IsValid())
        self.assertFalse(placeholder.GetUUIDString(), "the placeholder has no UUID")
        base = self.base_load_address(placeholder, target)
        self.assertNotEqual(base, lldb.LLDB_INVALID_ADDRESS)

        self.runCmd(
            "target modules replace --old-path /no/such/module.so '%s'" % replacement
        )

        new_module = target.FindModule(lldb.SBFileSpec(replacement))
        self.assertTrue(new_module.IsValid())
        self.assertEqual(self.base_load_address(new_module, target), base)
        self.assertFalse(
            target.FindModule(lldb.SBFileSpec("/no/such/module.so")).IsValid()
        )

    def test_failure_leaves_the_target_alone(self):
        """A replacement that cannot be placed is refused, and nothing is lost."""
        core = self.getBuildArtifact("no-uuid.dmp")
        self.yaml2obj("placeholder-no-uuid.yaml", core)

        target = self.dbg.CreateTarget("")
        self.assertTrue(target.LoadCore(core).IsValid())
        num_modules = target.GetNumModules()

        # An object file with no loadable segments cannot go where the
        # placeholder was.
        unplaceable = self.getBuildArtifact("unplaceable.o")
        self.yaml2obj("unplaceable.yaml", unplaceable)
        self.expect(
            "target modules replace --old-path /no/such/module.so '%s'" % unplaceable,
            error=True,
            substrs=["could not be loaded at"],
        )

        self.assertEqual(target.GetNumModules(), num_modules)
        self.assertTrue(
            target.FindModule(lldb.SBFileSpec("/no/such/module.so")).IsValid(),
            "the module that could not be replaced is still in the target",
        )

    @skipIfRemote
    @skipUnlessPlatform(["linux"])
    def test_core_file(self):
        """Replacing a module in a core file keeps it at the same address."""
        v1, v2, v1_copy = self.build_and_get_paths()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        target.BreakpointCreateBySourceRegex(
            "break after dlopen", lldb.SBFileSpec("main.cpp")
        )
        launch_info = target.GetLaunchInfo()
        launch_info.SetArguments([v1], True)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(error)

        core = self.getBuildArtifact("saved.core")
        self.runCmd("process save-core --style=full '%s'" % core)
        process.Kill()

        target = self.dbg.CreateTarget("")
        self.assertTrue(target.LoadCore(core).IsValid())
        old_module = target.FindModule(lldb.SBFileSpec(v1))
        self.assertTrue(old_module.IsValid())
        base = self.base_load_address(old_module, target)

        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)

        new_module = target.FindModule(lldb.SBFileSpec(v2))
        self.assertTrue(new_module.IsValid())
        self.assertEqual(self.base_load_address(new_module, target), base)
        self.assertTrue(new_module.FindSymbol("only_in_v2").IsValid())

    @skipIfRemote
    def test_breakpoints_move_to_the_replacement(self):
        """Breakpoint locations are re-resolved into the new module."""
        v1, v2, v1_copy = self.build_and_get_paths()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        target.BreakpointCreateBySourceRegex(
            "break after dlopen", lldb.SBFileSpec("main.cpp")
        )

        launch_info = target.GetLaunchInfo()
        launch_info.SetArguments([v1], True)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(error, "the process launched")
        self.assertState(process.GetState(), lldb.eStateStopped)

        old_module = target.FindModule(lldb.SBFileSpec(v1))
        self.assertTrue(old_module.IsValid(), "v1 was dlopen'd")
        old_uuid = old_module.GetUUIDString()

        # A breakpoint on a symbol both variants define, and one on a symbol
        # only the old variant defines.
        common_bp = target.BreakpointCreateByName("common_func")
        self.assertEqual(common_bp.GetNumLocations(), 1)
        only_v1_bp = target.BreakpointCreateByName("only_in_v1")
        self.assertEqual(only_v1_bp.GetNumLocations(), 1)

        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)

        # The shared symbol re-resolves, and nothing still points into the
        # module that was removed.
        self.assertGreaterEqual(common_bp.GetNumLocations(), 1)
        for i in range(common_bp.GetNumLocations()):
            module = common_bp.GetLocationAtIndex(i).GetAddress().GetModule()
            self.assertNotEqual(
                module.GetUUIDString(),
                old_uuid,
                "no location still resolves into the replaced module",
            )

        # The symbol that only existed in the old variant goes pending.
        self.assertEqual(only_v1_bp.GetNumLocations(), 0)

    @skipIfRemote
    @skipUnlessPlatform(["linux"])
    @skipIf(archs=no_match(["x86_64"]))
    def test_thread_local_storage_still_resolves(self):
        """The dynamic loader's per module state follows the replacement.

        The loader keys the link map address it needs for TLS lookups off the
        module itself, so without help the replacement has no link map and every
        thread local in it reads back as "no TLS data currently exists"."""
        v1, v2, v1_copy = self.build_and_get_paths()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        target.BreakpointCreateBySourceRegex(
            "break after dlopen", lldb.SBFileSpec("main.cpp")
        )

        launch_info = target.GetLaunchInfo()
        launch_info.SetArguments([v1], True)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(error, "the process launched")
        self.assertState(process.GetState(), lldb.eStateStopped)

        # Sanity check that TLS resolves at all before the replace, so that a
        # failure below is attributable to the replace and not to the platform.
        before = target.EvaluateExpression("tls_var")
        self.assertSuccess(before.GetError(), "TLS resolves before the replace")
        self.assertEqual(before.GetValueAsSigned(), 701)

        self.runCmd("target modules replace --allow-uuid-mismatch --force '%s'" % v2)

        after = target.EvaluateExpression("tls_var")
        self.assertSuccess(after.GetError(), "TLS still resolves after the replace")
        # The variable is read out of the live process, whose mapped pages are
        # still the old library's, so the value is v1's. What matters is that the
        # lookup resolves at all instead of failing to find a link map.
        self.assertNotEqual(after.GetValueAsSigned(), 0)
