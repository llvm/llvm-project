import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import (
    requireExpressionEvaluation,
    skipUnlessPlatform,
)
from lldbsuite.test.lldbtest import TestBase, line_number


@skipUnlessPlatform(["linux"])
@requireExpressionEvaluation
class OverriddenWeakSymbolsTest(TestBase):
    SHARED_BUILD_TESTCASE = False

    def check_functions(self, cxx, reverse, pie, both_weak=False):
        extension = "cpp" if cxx else "c"
        prefix = "cxx-" if cxx else ""
        sources = [f"{prefix}main.{extension}", f"{prefix}weak.{extension}"]
        if reverse:
            sources.reverse()
        cflags = "-fPIE" if pie else "-fno-pie"
        if both_weak:
            cflags += " -DBOTH_WEAK"
        self.build(
            dictionary={
                "C_SOURCES": "" if cxx else " ".join(sources),
                "CXX_SOURCES": " ".join(sources) if cxx else "",
                "CFLAGS_EXTRAS": cflags,
                "LD_EXTRAS": "-pie" if pie else "-no-pie",
            }
        )
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "weak_context"
        )
        # Check both return value (11 vs 22) and return type (StrongResult vs
        # WeakResult) to verify that the linked definition's DIE was selected.
        self.expect_expr("action()", result_type="StrongResult", result_value="11")
        self.expect_expr("other_action()", result_value="13")
        self.expect_expr("weak_only()", result_value="7")
        self.expect_expr("local_only()", result_value="77")
        self.expect_expr("scoped_action()", result_value="77")
        self.expect_expr("value", result_value="11")

        pointer_breakpoint = target.BreakpointCreateByLocation(
            "weak.c", line_number("weak.c", "lookup in pointer scope")
        )
        self.assertEqual(pointer_breakpoint.GetNumLocations(), 1)
        process.Continue()
        self.expect_expr("action()", result_value="77")
        self.expect_expr("value", result_value="99")
        target.BreakpointDelete(pointer_breakpoint.GetID())

        breakpoint = target.BreakpointCreateByName("main_context")
        self.assertEqual(breakpoint.GetNumLocations(), 1)
        process.Continue()
        self.expect_expr("observed_action", result_value="11")
        self.expect_expr("observed_other", result_value="13")
        self.expect_expr("action()", result_type="StrongResult", result_value="11")
        self.expect_expr("other_action()", result_value="13")
        self.expect_expr("&action == program_action", result_value="true")
        self.expect_expr("&value == program_value", result_value="true")
        self.expect_expr("scoped_action(0)", result_value="66")
        if not cxx:
            # The overridden body in main.c is still callable via its alias.
            self.expect_expr("(int)retained_action()", result_value="22")
        if cxx:
            self.expect_expr("ns::selected()", result_value="33")
            self.expect_expr("ns::selected(41)", result_value="42")
            self.expect_expr("value_ref", result_value="11")
            self.expect_expr("&value_ref == program_value", result_value="true")
            self.expect_expr("value_ref = 17", result_value="17")
            self.expect_expr("*program_value", result_value="17")

        # The retained body remains available to an explicit address query.
        matches = target.FindFunctions("action", lldb.eFunctionNameTypeBase)
        for context in matches:
            function = context.GetFunction()
            if (
                function.IsValid()
                and context.GetCompileUnit().GetFileSpec().GetFilename()
                == f"{prefix}main.{extension}"
                and function.GetName() in ("action", "action()")
            ):
                address = function.GetStartAddress().GetLoadAddress(target)
                self.expect_expr(f"((int (*)(void)){address})()", result_value="22")
                break
        else:
            self.fail("Retained weak function is missing")

    def test_c(self):
        self.check_functions(cxx=False, reverse=False, pie=False)

    def test_c_reverse_pie(self):
        self.check_functions(cxx=False, reverse=True, pie=True)

    def test_cxx(self):
        self.check_functions(cxx=True, reverse=False, pie=False)

    def test_cxx_reverse_pie(self):
        self.check_functions(cxx=True, reverse=True, pie=True)

    def test_two_weak_definitions(self):
        # The selected ELF definition can itself be weak.
        self.check_functions(cxx=False, reverse=True, pie=False, both_weak=True)

    def test_selected_function_without_debug_info(self):
        self.build(dictionary={"C_SOURCES": "no-debug-main.c no-debug-strong.c"})
        lldbutil.run_to_name_breakpoint(self, "context")
        # The visible declaration still supplies the type when the linked
        # definition has no debug info.
        self.expect_expr("action()", result_type="Result", result_value="11")
        self.expect_expr("(int)action()", result_value="11")
        self.expect_expr("program_action()", result_value="11")
        self.expect_expr("&action == program_action", result_value="true")

    def test_symbol_only_call(self):
        self.build(dictionary={"C_SOURCES": "symbol-only.c"})
        lldbutil.run_to_name_breakpoint(self, "context")
        self.expect_expr("(int)action()", result_value="11")

    def test_shared_library_override(self):
        self.build(
            dictionary={
                "C_SOURCES": "shared-main.c shared-strong.c",
                "DYLIB_C_SOURCES": "shared-weak.c",
                "DYLIB_NAME": "weak",
                "LD_EXTRAS": "-Wl,--export-dynamic",
            }
        )
        lldbutil.run_to_name_breakpoint(
            self, "main_context", extra_images=["libweak.so"]
        )
        # Only the shared library provides a function type, but the executable
        # overrides that definition. Do not pin calls to the type's module.
        self.expect_expr("action()", result_value="11")
        self.expect_expr("observed_action", result_value="11")
        self.expect_expr("program_action()", result_value="11")
        self.expect_expr("&action == program_action", result_value="true")

    def test_localized_function(self):
        self.build(
            dictionary={
                "C_SOURCES": "localized-main.c",
                "OBJECTS": "localized.o localized-main.o",
            }
        )
        lldbutil.run_to_name_breakpoint(self, "local_context")
        # `objcopy --localize-symbol` turns `action` into a local ELF symbol in
        # localized.o while its DWARF DIE still has DW_AT_external. Evaluating
        # `action()` in `local_context` must call the local definition (22), not
        # the global one (11).
        self.expect_expr("action()", result_value="22")
        self.expect_expr("local_action()", result_value="22")
        self.expect_expr("global_action()", result_value="11")
        self.expect_expr("&action == local_action", result_value="true")
