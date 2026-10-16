"""Keep primary JIT linkage lookup separate from compatibility aliases."""

import lldb
from lldbsuite.test.decorators import skipUnlessDarwin
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestIRExecutionUnitLinkageLookup(TestBase):
    def launch(self):
        self.build()
        self.runCmd("settings set target.preload-symbols false")
        self.runCmd("settings set target.load-script-from-symbol-file false")
        self.target, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.frame = thread.frames[0]

    def value(self, expression, declarations=""):
        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        options.SetPrefix(declarations)
        result = self.frame.EvaluateExpression(expression, options)
        self.assertSuccess(result.GetError())
        return result.GetValueAsSigned()

    @skipUnlessDarwin
    def test_primary_c_linkage_ignores_symbol_only_cpp_decoy(self):
        self.launch()
        self.assertEqual(
            self.value("linkage_target()", 'extern "C" int linkage_target();'),
            42,
        )

    @skipUnlessDarwin
    def test_global_and_weak_symbols(self):
        self.launch()
        self.assertEqual(
            self.value("linkage_global", 'extern "C" { extern int linkage_global; }'),
            73,
        )
        self.assertEqual(
            self.value(
                "missing_weak == nullptr",
                'extern "C" int missing_weak() __attribute__((weak_import));',
            ),
            1,
        )
        self.assertEqual(
            self.value(
                "present_weak()",
                'extern "C" int present_weak() __attribute__((weak_import));',
            ),
            29,
        )

    @skipUnlessDarwin
    def test_cpp_and_function_call_label_fallbacks(self):
        self.launch()
        self.assertEqual(
            self.value("cpp_fallback_target()", "int cpp_fallback_target();"), 17
        )
        self.assertEqual(
            self.value(
                "labelled_fallback()",
                'int labelled_fallback() asm("$__lldb_func:generic:'
                '0xffffffffffffffff:0:fcl_alias_target");',
            ),
            19,
        )

    @skipUnlessDarwin
    def test_dwarf_callable(self):
        self.launch()
        self.assertEqual(self.value("dwarf_callable()"), 31)
