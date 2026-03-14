"""
Test platform locate module callback functionality
"""

import ctypes
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from pathlib import Path

import lldb

UNITTESTS_TARGET_INPUTS_PATH = "../../../../unittests/Target/Inputs"
MODULE_PLATFORM_PATH = "/system/lib64/AndroidModule.so"
MODULE_TRIPLE = "aarch64-none-linux"
MODULE_RESOLVED_TRIPLE = "aarch64--linux-android"
MODULE_UUID = "80008338-82A0-51E5-5922-C905D23890DA-BDDEFECC"
MODULE_FUNCTION = "boom"
MODULE_HIDDEN_FUNCTION = "boom_hidden"
MODULE_FILE = "AndroidModule.so"
MODULE_NON_EXISTENT_FILE = "non-existent-file"
SYMBOL_FILE = "AndroidModule.unstripped.so"
BREAKPAD_SYMBOL_FILE = "AndroidModule.so.sym"
SYMBOL_STRIPPED = "stripped"
SYMBOL_UNSTRIPPED = "unstripped"


class LocateModuleCallbackTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.platform = self.dbg.GetSelectedPlatform()
        self.target = self.dbg.CreateTarget("")
        self.assertTrue(self.target)

        self.input_dir = (
            Path(self.getSourceDir()) / UNITTESTS_TARGET_INPUTS_PATH
        ).resolve()
        self.assertTrue(self.input_dir.is_dir())

    def check_module_spec(self, module_spec: lldb.SBModuleSpec):
        self.assertEqual(
            MODULE_UUID.replace("-", ""),
            ctypes.string_at(
                int(module_spec.GetUUIDBytes()),
                module_spec.GetUUIDLength(),
            )
            .hex()
            .upper(),
        )

        self.assertEqual(MODULE_TRIPLE, module_spec.GetTriple())

        self.assertEqual(
            MODULE_PLATFORM_PATH, Path(module_spec.GetFileSpec().fullpath).as_posix()
        )

    def check_module(self, module: lldb.SBModule, symbol_file: str, symbol_kind: str):
        self.assertTrue(module.IsValid())

        self.assertEqual(
            MODULE_UUID,
            module.GetUUIDString(),
        )

        self.assertEqual(MODULE_RESOLVED_TRIPLE, module.GetTriple())

        self.assertEqual(
            MODULE_PLATFORM_PATH, Path(module.GetPlatformFileSpec().fullpath).as_posix()
        )

        self.assertTrue(
            (self.input_dir / MODULE_FILE)
            .resolve()
            .samefile(Path(module.GetFileSpec().fullpath).resolve())
        )

        self.assertTrue(
            (self.input_dir / symbol_file)
            .resolve()
            .samefile(Path(module.GetSymbolFileSpec().fullpath).resolve())
        )

        sc_list = module.FindFunctions(MODULE_FUNCTION, lldb.eSymbolTypeCode)
        self.assertEqual(1, sc_list.GetSize())
        sc_list = module.FindFunctions(MODULE_HIDDEN_FUNCTION, lldb.eSymbolTypeCode)
        self.assertEqual(0 if symbol_kind == SYMBOL_STRIPPED else 1, sc_list.GetSize())

    def test_set_non_callable(self):
        # The callback should be callable.
        non_callable = "a"

        with self.assertRaises(TypeError, msg="Need a callable object or None!"):
            self.platform.SetLocateModuleCallback(non_callable)

    def test_set_wrong_args(self):
        # The callback should accept 3 argument.
        def test_args2(a, b):
            pass

        with self.assertRaises(TypeError, msg="Expected 3 argument callable object"):
            self.platform.SetLocateModuleCallback(test_args2)

    def test_default(self):
        # The default behavior is to locate the module with LLDB implementation
        # and AddModule should fail.
        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.assertFalse(module)

    def test_set_none(self):
        # SetLocateModuleCallback should succeed to clear the callback with None.
        # and AddModule should fail.
        self.assertTrue(self.platform.SetLocateModuleCallback(None).Success())

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.assertFalse(module)

    def test_return_error(self):
        # The callback fails, AddModule should fail.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)
            return lldb.SBError("locate module callback failed")

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.assertFalse(module)

    def test_return_no_files(self):
        # The callback succeeds but not return any files, AddModule should fail.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)
            return lldb.SBError()

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.assertFalse(module)

    def test_return_non_existent_module(self):
        # The callback returns non-existent module file, AddModule should fail.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)

            module_file_spec.SetDirectory(str(self.input_dir))
            module_file_spec.SetFilename(MODULE_NON_EXISTENT_FILE)

            return lldb.SBError()

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.assertFalse(module)

    def test_return_module_with_non_existent_symbol(self):
        # The callback returns a module and non-existent symbol file,
        # AddModule should fail.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)

            module_file_spec.SetDirectory(str(self.input_dir))
            module_file_spec.SetFilename(MODULE_FILE)

            symbol_file_spec.SetDirectory(str(self.input_dir))
            symbol_file_spec.SetFilename(MODULE_NON_EXISTENT_FILE)

            return lldb.SBError()

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.assertFalse(module)

    def test_return_non_existent_symbol(self):
        # The callback returns non-existent symbol file, AddModule should fail.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)

            symbol_file_spec.SetDirectory(str(self.input_dir))
            symbol_file_spec.SetFilename(MODULE_NON_EXISTENT_FILE)

            return lldb.SBError()

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.assertFalse(module)

    def test_return_module(self):
        # The callback returns the module file, AddModule should succeed.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)

            module_file_spec.SetDirectory(str(self.input_dir))
            module_file_spec.SetFilename(MODULE_FILE)

            return lldb.SBError()

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.check_module(
            module=module, symbol_file=MODULE_FILE, symbol_kind=SYMBOL_STRIPPED
        )

    def test_return_module_with_symbol(self):
        # The callback returns the module file and the symbol file,
        # AddModule should succeed.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)

            module_file_spec.SetDirectory(str(self.input_dir))
            module_file_spec.SetFilename(MODULE_FILE)

            symbol_file_spec.SetDirectory(str(self.input_dir))
            symbol_file_spec.SetFilename(SYMBOL_FILE)

            return lldb.SBError()

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.check_module(
            module=module, symbol_file=SYMBOL_FILE, symbol_kind=SYMBOL_UNSTRIPPED
        )

    def test_return_module_with_breakpad_symbol(self):
        # The callback returns the module file and the breakpad symbol file,
        # AddModule should succeed.
        def test_locate_module(
            module_spec: lldb.SBModuleSpec,
            module_file_spec: lldb.SBFileSpec,
            symbol_file_spec: lldb.SBFileSpec,
        ):
            self.check_module_spec(module_spec)

            module_file_spec.SetDirectory(str(self.input_dir))
            module_file_spec.SetFilename(MODULE_FILE)

            symbol_file_spec.SetDirectory(str(self.input_dir))
            symbol_file_spec.SetFilename(BREAKPAD_SYMBOL_FILE)

            return lldb.SBError()

        self.assertTrue(
            self.platform.SetLocateModuleCallback(test_locate_module).Success()
        )

        module = self.target.AddModule(
            MODULE_PLATFORM_PATH,
            MODULE_TRIPLE,
            MODULE_UUID,
        )

        self.check_module(
            module=module,
            symbol_file=BREAKPAD_SYMBOL_FILE,
            symbol_kind=SYMBOL_UNSTRIPPED,
        )
