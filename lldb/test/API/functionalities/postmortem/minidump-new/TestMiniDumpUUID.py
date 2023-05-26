"""
Test basics of Minidump debugging.
"""

import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiniDumpUUIDTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def verify_module(self, module, verify_path, verify_uuid):
        # Compare the filename and the directory separately. We are avoiding
        # SBFileSpec.fullpath because it causes a slash/backslash confusion
        # on Windows.  Similarly, we compare the directories using normcase
        # because they may contain a Linux-style relative path from the
        # minidump appended to a Windows-style root path from the host.
        self.assertEqual(os.path.basename(verify_path), module.GetFileSpec().basename)
        self.assertEqual(
            os.path.normcase(os.path.dirname(verify_path)),
            os.path.normcase(module.GetFileSpec().dirname or ""),
        )
        self.assertEqual(verify_uuid, module.GetUUIDString())

    def get_minidump_modules(self, yaml_file, exe=None):
        minidump_path = self.getBuildArtifact(os.path.basename(yaml_file) + ".dmp")
        self.yaml2obj(yaml_file, minidump_path)
        self.target = self.dbg.CreateTarget(exe)
        self.process = self.target.LoadCore(minidump_path)
        return self.target.modules

    def test_zero_uuid_modules(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is valid,
        but contains a PDB70 value whose age is zero and whose UUID values are
        all zero. Prior to a fix all such modules would be duplicated to the
        first one since the UUIDs claimed to be valid and all zeroes. Now we
        ensure that the UUID is not valid for each module and that we have
        each of the modules in the target after loading the core
        """
        modules = self.get_minidump_modules("linux-arm-zero-uuids.yaml")
        self.assertEqual(2, len(modules))
        self.verify_module(modules[0], "/file/does/not/exist/a", None)
        self.verify_module(modules[1], "/file/does/not/exist/b", None)

    def test_uuid_modules_no_age(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is valid,
        and contains a PDB70 value whose age is zero and whose UUID values are
        valid. Ensure we decode the UUID and don't include the age field in the UUID.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-no-age.yaml")
        modules = self.target.modules
        self.assertEqual(2, len(modules))
        self.verify_module(modules[0], "/tmp/a", "01020304-0506-0708-090A-0B0C0D0E0F10")
        self.verify_module(modules[1], "/tmp/b", "0A141E28-323C-4650-5A64-6E78828C96A0")

    def test_uuid_modules_no_age_apple(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is valid,
        and contains a PDB70 value whose age is zero and whose UUID values are
        valid. Ensure we decode the UUID and don't include the age field in the UUID.
        Also ensure that the first uint32_t is byte swapped, along with the next
        two uint16_t values. Breakpad incorrectly byte swaps these values when it
        saves Darwin minidump files.
        """
        modules = self.get_minidump_modules("macos-arm-uuids-no-age.yaml")
        modules = self.target.modules
        self.assertEqual(2, len(modules))
        self.verify_module(modules[0], "/tmp/a", "04030201-0605-0807-090A-0B0C0D0E0F10")
        self.verify_module(modules[1], "/tmp/b", "281E140A-3C32-5046-5A64-6E78828C96A0")

    def test_uuid_modules_with_age(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is valid,
        and contains a PDB70 value whose age is valid and whose UUID values are
        valid. Ensure we decode the UUID and include the age field in the UUID.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-with-age.yaml")
        self.assertEqual(2, len(modules))
        self.verify_module(
            modules[0], "/tmp/a", "01020304-0506-0708-090A-0B0C0D0E0F10-10101010"
        )
        self.verify_module(
            modules[1], "/tmp/b", "0A141E28-323C-4650-5A64-6E78828C96A0-20202020"
        )

    def test_uuid_modules_elf_build_id_16(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is valid,
        and contains a ELF build ID whose value is valid and is 16 bytes long.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-elf-build-id-16.yaml")
        self.assertEqual(2, len(modules))
        self.verify_module(modules[0], "/tmp/a", "01020304-0506-0708-090A-0B0C0D0E0F10")
        self.verify_module(modules[1], "/tmp/b", "0A141E28-323C-4650-5A64-6E78828C96A0")

    def test_uuid_modules_elf_build_id_20(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is valid,
        and contains a ELF build ID whose value is valid and is 20 bytes long.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-elf-build-id-20.yaml")
        self.assertEqual(2, len(modules))
        self.verify_module(
            modules[0], "/tmp/a", "01020304-0506-0708-090A-0B0C0D0E0F10-11121314"
        )
        self.verify_module(
            modules[1], "/tmp/b", "0A141E28-323C-4650-5A64-6E78828C96A0-AAB4BEC8"
        )

    def test_uuid_modules_elf_build_id_zero(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is valid,
        and contains a ELF build ID whose value is all zero.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-elf-build-id-zero.yaml")
        self.assertEqual(2, len(modules))
        self.verify_module(modules[0], "/not/exist/a", None)
        self.verify_module(modules[1], "/not/exist/b", None)

    def test_uuid_modules_elf_build_id_same(self):
        """
        Test multiple modules having a MINIDUMP_MODULE.CvRecord that is
        valid, and contains a ELF build ID whose value is the same. There
        is an assert in the PlaceholderObjectFile that was firing when we
        encountered this which was crashing the process that was checking
        if PlaceholderObjectFile.m_base was the same as the address this
        fake module was being loaded at. We need to ensure we don't crash
        in such cases and that we add both modules even though they have
        the same UUID.
        """
        modules = self.get_minidump_modules("linux-arm-same-uuids.yaml")
        self.assertEqual(2, len(modules))
        self.verify_module(
            modules[0],
            "/file/does/not/exist/a",
            "11223344-1122-3344-1122-334411223344-11223344",
        )
        self.verify_module(
            modules[1],
            "/file/does/not/exist/b",
            "11223344-1122-3344-1122-334411223344-11223344",
        )

    def test_partial_uuid_match(self):
        """
        Breakpad has been known to create minidump files using CvRecord in each
        module whose signature is set to PDB70 where the UUID only contains the
        first 16 bytes of a 20 byte ELF build ID. Code was added to
        ProcessMinidump.cpp to deal with this and allows partial UUID matching.

        This test verifies that if we have a minidump with a 16 byte UUID, that
        we are able to associate a symbol file with a 20 byte UUID only if the
        first 16 bytes match. In this case we will see the path from the file
        we found in the test directory and the 20 byte UUID from the actual
        file, not the 16 byte shortened UUID from the minidump.
        """
        so_path = self.getBuildArtifact("libuuidmatch.so")
        self.yaml2obj("libuuidmatch.yaml", so_path)
        cmd = 'settings set target.exec-search-paths "%s"' % (os.path.dirname(so_path))
        self.dbg.HandleCommand(cmd)
        modules = self.get_minidump_modules("linux-arm-partial-uuids-match.yaml")
        self.assertEqual(1, len(modules))
        self.verify_module(
            modules[0], so_path, "7295E17C-6668-9E05-CBB5-DEE5003865D5-5267C116"
        )

    def test_partial_uuid_mismatch(self):
        """
        Breakpad has been known to create minidump files using CvRecord in each
        module whose signature is set to PDB70 where the UUID only contains the
        first 16 bytes of a 20 byte ELF build ID. Code was added to
        ProcessMinidump.cpp to deal with this and allows partial UUID matching.

        This test verifies that if we have a minidump with a 16 byte UUID, that
        we are not able to associate a symbol file with a 20 byte UUID only if
        any of the first 16 bytes do not match. In this case we will see the UUID
        from the minidump file and the path from the minidump file.
        """
        so_path = self.getBuildArtifact("libuuidmismatch.so")
        self.yaml2obj("libuuidmismatch.yaml", so_path)
        cmd = 'settings set target.exec-search-paths "%s"' % (os.path.dirname(so_path))
        self.dbg.HandleCommand(cmd)
        modules = self.get_minidump_modules("linux-arm-partial-uuids-mismatch.yaml")
        self.assertEqual(1, len(modules))
        self.verify_module(
            modules[0],
            "/invalid/path/on/current/system/libuuidmismatch.so",
            "7295E17C-6668-9E05-CBB5-DEE5003865D5",
        )

    def test_breakpad_hash_match(self):
        """
        Breakpad creates minidump files using CvRecord in each module whose
        signature is set to PDB70 where the UUID is a hash generated by
        breakpad of the .text section. This is only done when the
        executable has no ELF build ID.

        This test verifies that if we have a minidump with a 16 byte UUID,
        that we are able to associate a symbol file with no ELF build ID
        and match it up by hashing the .text section.
        """
        so_path = self.getBuildArtifact("libbreakpad.so")
        self.yaml2obj("libbreakpad.yaml", so_path)
        cmd = 'settings set target.exec-search-paths "%s"' % (os.path.dirname(so_path))
        self.dbg.HandleCommand(cmd)
        modules = self.get_minidump_modules("linux-arm-breakpad-uuid-match.yaml")
        self.assertEqual(1, len(modules))
        # LLDB makes up it own UUID as well when there is no build ID so we
        # will check that this matches.
        self.verify_module(modules[0], so_path, "D9C480E8")

    def test_breakpad_hash_match_sysroot(self):
        """
        Check that we can match the breakpad .text section hash when the
        module is located under a user-provided sysroot.
        """
        sysroot_path = os.path.join(self.getBuildDir(), "mock_sysroot")
        # Create the directory under the sysroot where the minidump reports
        # the module.
        so_dir = os.path.join(
            sysroot_path, "invalid", "path", "on", "current", "system"
        )
        so_path = os.path.join(so_dir, "libbreakpad.so")
        lldbutil.mkdir_p(so_dir)
        self.yaml2obj("libbreakpad.yaml", so_path)
        self.runCmd("platform select remote-linux --sysroot '%s'" % sysroot_path)
        modules = self.get_minidump_modules("linux-arm-breakpad-uuid-match.yaml")
        self.assertEqual(1, len(modules))
        # LLDB makes up its own UUID as well when there is no build ID so we
        # will check that this matches.
        self.verify_module(modules[0], so_path, "D9C480E8")

    def test_breakpad_hash_match_sysroot_decoy(self):
        """
        Check that we can match the breakpad .text section hash when there is
        a module with the right name but wrong contents under a user-provided
        sysroot, and the right module is at the given search path..
        """
        sysroot_path = os.path.join(self.getBuildDir(), "mock_sysroot")
        # Create the directory under the sysroot where the minidump reports
        # the module.
        decoy_dir = os.path.join(
            sysroot_path, "invalid", "path", "on", "current", "system"
        )
        decoy_path = os.path.join(decoy_dir, "libbreakpad.so")
        lldbutil.mkdir_p(decoy_dir)
        self.yaml2obj("libbreakpad-decoy.yaml", decoy_path)
        self.runCmd("platform select remote-linux --sysroot '%s'" % sysroot_path)
        so_dir = os.path.join(self.getBuildDir(), "searchpath_dir")
        so_path = os.path.join(so_dir, "libbreakpad.so")
        lldbutil.mkdir_p(so_dir)
        self.yaml2obj("libbreakpad.yaml", so_path)
        self.runCmd('settings set target.exec-search-paths "%s"' % so_dir)
        modules = self.get_minidump_modules("linux-arm-breakpad-uuid-match.yaml")
        self.assertEqual(1, len(modules))
        # LLDB makes up its own UUID as well when there is no build ID so we
        # will check that this matches.
        self.verify_module(modules[0], so_path, "D9C480E8")

    def test_breakpad_overflow_hash_match(self):
        """
        This is a similar to test_breakpad_hash_match, but it verifies that
        if the .text section does not end on a 16 byte boundary, then it
        will overflow into the next section's data by up to 15 bytes. This
        verifies that we are able to match what breakpad does as it will do
        this.
        """
        so_path = self.getBuildArtifact("libbreakpad.so")
        self.yaml2obj("libbreakpad-overflow.yaml", so_path)
        cmd = 'settings set target.exec-search-paths "%s"' % (os.path.dirname(so_path))
        self.dbg.HandleCommand(cmd)
        modules = self.get_minidump_modules("linux-arm-breakpad-uuid-match.yaml")
        self.assertEqual(1, len(modules))
        # LLDB makes up it own UUID as well when there is no build ID so we
        # will check that this matches.
        self.verify_module(modules[0], so_path, "48EB9FD7")

    def test_breakpad_hash_match_exe_outside_sysroot(self):
        """
        Check that we can match the breakpad .text section hash when the
        module is specified as the exe during launch, and a syroot is
        provided, which does not contain the exe.
        """
        sysroot_path = os.path.join(self.getBuildDir(), "mock_sysroot")
        lldbutil.mkdir_p(sysroot_path)
        so_dir = os.path.join(self.getBuildDir(), "binary")
        so_path = os.path.join(so_dir, "libbreakpad.so")
        lldbutil.mkdir_p(so_dir)
        self.yaml2obj("libbreakpad.yaml", so_path)
        self.runCmd("platform select remote-linux --sysroot '%s'" % sysroot_path)
        modules = self.get_minidump_modules(
            "linux-arm-breakpad-uuid-match.yaml", so_path
        )
        self.assertEqual(1, len(modules))
        # LLDB makes up its own UUID as well when there is no build ID so we
        # will check that this matches.
        self.verify_module(modules[0], so_path, "D9C480E8")

    def test_facebook_hash_match(self):
        """
        Breakpad creates minidump files using CvRecord in each module whose
        signature is set to PDB70 where the UUID is a hash generated by
        breakpad of the .text section and Facebook modified this hash to
        avoid collisions. This is only done when the executable has no ELF
        build ID.

        This test verifies that if we have a minidump with a 16 byte UUID,
        that we are able to associate a symbol file with no ELF build ID
        and match it up by hashing the .text section like Facebook does.
        """
        so_path = self.getBuildArtifact("libbreakpad.so")
        self.yaml2obj("libbreakpad.yaml", so_path)
        cmd = 'settings set target.exec-search-paths "%s"' % (os.path.dirname(so_path))
        self.dbg.HandleCommand(cmd)
        modules = self.get_minidump_modules("linux-arm-facebook-uuid-match.yaml")
        self.assertEqual(1, len(modules))
        # LLDB makes up it own UUID as well when there is no build ID so we
        # will check that this matches.
        self.verify_module(modules[0], so_path, "D9C480E8")

    def test_relative_module_name(self):
        old_cwd = os.getcwd()
        self.addTearDownHook(lambda: os.chdir(old_cwd))
        os.chdir(self.getBuildDir())
        name = "file-with-a-name-unlikely-to-exist-in-the-current-directory.so"
        open(name, "a").close()
        modules = self.get_minidump_modules(
            self.getSourcePath("relative_module_name.yaml")
        )
        self.assertEqual(1, len(modules))
        self.verify_module(modules[0], name, None)

    def test_add_module_build_id_16(self):
        """
        Test that adding module with 16 byte UUID returns the existing
        module or fails.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-elf-build-id-16.yaml")
        self.assertEqual(2, len(modules))

        # Add the existing modules.
        self.assertEqual(
            modules[0],
            self.target.AddModule(
                "/some/local/a", "", "01020304-0506-0708-090A-0B0C0D0E0F10"
            ),
        )
        self.assertEqual(
            modules[1],
            self.target.AddModule(
                "/some/local/b", "", "0A141E28-323C-4650-5A64-6E78828C96A0"
            ),
        )

        # Adding modules with non-existing UUID should fail.
        self.assertFalse(
            self.target.AddModule(
                "a", "", "12345678-1234-1234-1234-123456789ABC"
            ).IsValid()
        )
        self.assertFalse(
            self.target.AddModule(
                "a", "", "01020304-0506-0708-090A-0B0C0D0E0F10-12345678"
            ).IsValid()
        )

    def test_add_module_build_id_20(self):
        """
        Test that adding module with 20 byte UUID returns the existing
        module or fails.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-elf-build-id-20.yaml")

        # Add the existing modules.
        self.assertEqual(
            modules[0],
            self.target.AddModule(
                "/some/local/a", "", "01020304-0506-0708-090A-0B0C0D0E0F10-11121314"
            ),
        )
        self.assertEqual(
            modules[1],
            self.target.AddModule(
                "/some/local/b", "", "0A141E28-323C-4650-5A64-6E78828C96A0-AAB4BEC8"
            ),
        )

        # Adding modules with non-existing UUID should fail.
        self.assertFalse(
            self.target.AddModule(
                "a", "", "01020304-0506-0708-090A-0B0C0D0E0F10"
            ).IsValid()
        )
        self.assertFalse(
            self.target.AddModule(
                "a", "", "01020304-0506-0708-090A-0B0C0D0E0F10-12345678"
            ).IsValid()
        )

    def test_add_module_build_id_4(self):
        """
        Test that adding module with 4 byte UUID returns the existing
        module or fails.
        """
        modules = self.get_minidump_modules("linux-arm-uuids-elf-build-id-4.yaml")

        # Add the existing modules.
        self.assertEqual(
            modules[0], self.target.AddModule("/some/local/a.so", "", "01020304")
        )
        self.assertEqual(
            modules[1], self.target.AddModule("/some/local/b.so", "", "0A141E28")
        )

        # Adding modules with non-existing UUID should fail.
        self.assertFalse(
            self.target.AddModule(
                "a", "", "01020304-0506-0708-090A-0B0C0D0E0F10"
            ).IsValid()
        )
        self.assertFalse(self.target.AddModule("a", "", "01020305").IsValid())

    def test_remove_placeholder_add_real_module(self):
        """
        Test that removing a placeholder module and adding back the real
        module succeeds.
        """
        so_path = self.getBuildArtifact("libuuidmatch.so")
        self.yaml2obj("libuuidmatch.yaml", so_path)
        modules = self.get_minidump_modules("linux-arm-uuids-match.yaml")

        uuid = "7295E17C-6668-9E05-CBB5-DEE5003865D5-5267C116"
        self.assertEqual(1, len(modules))
        self.verify_module(modules[0], "/target/path/libuuidmatch.so", uuid)

        self.target.RemoveModule(modules[0])
        new_module = self.target.AddModule(so_path, "", uuid)

        self.verify_module(new_module, so_path, uuid)
        self.assertEqual(new_module, self.target.modules[0])
        self.assertEqual(1, len(self.target.modules))
