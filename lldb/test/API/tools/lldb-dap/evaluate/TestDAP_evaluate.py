"""
Test lldb-dap evaluate request
"""

from typing import Optional

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.session_helpers import ExpectEval, FrameContext
from lldbsuite.test.tools.lldb_dap.types import EvaluateContext, LaunchArgs, ValueFormat


class TestDAP_evaluate(DAPTestCaseBase):
    # The frame that `assert_eval*` functions calls evaluate in.
    _eval_frame: Optional[FrameContext] = None

    def set_evaluation_frame(self, frame: Optional[FrameContext]):
        """Set the frame that `assert_eval*` functions calls evaluate in."""
        self._eval_frame = frame

    def run_evaluate_expressions(
        self,
        context: Optional[EvaluateContext] = None,
        enableAutoVariableSummaries: bool = False,
    ):
        self.set_evaluation_frame(None)
        is_result_expanded = context == "repl"
        is_result_brief = context == "clipboard"
        is_result_summary = (
            not is_result_expanded
            and not is_result_brief
            and enableAutoVariableSummaries
        )
        context_parses_expressions = context != "hover"
        session = self.build_and_create_session()

        def assert_eval(expression: str, matches: str, *, as_hex=False, **expects):
            fmt = ValueFormat(hex=True) if as_hex else None
            if eval_frame := self._eval_frame:
                body = eval_frame.evaluate(expression, context=context, format=fmt)
            else:
                body = session.evaluate(expression, context=context, format=fmt)
            expects.setdefault("has_mem_ref", True)
            session.verify_evaluate(body, ExpectEval(matches=matches, **expects))
            return body

        def assert_eval_fails(expression: str):
            frame_id = self._eval_frame.id if self._eval_frame else None
            session.do_evaluate(expression, frameId=frame_id, context=context).error(
                f"expected {expression!r} to fail using {context=!r} in {frame_id=!r}"
            )

        source = "main.cpp"
        program = self.getBuildArtifact("a.out")
        breakpoint_lines = [
            line_number(source, f"// breakpoint 1"),
            line_number(source, f"// breakpoint 2"),
            line_number(source, f"// breakpoint 3"),
            line_number(source, f"// breakpoint 4"),
            line_number(source, f"// breakpoint 5"),
            line_number(source, f"// breakpoint 6"),
            line_number(source, f"// breakpoint 7"),
            line_number(source, f"// breakpoint 8"),
        ]

        launch = LaunchArgs(
            program, enableAutoVariableSummaries=enableAutoVariableSummaries
        )
        with session.configure(launch) as cfg:
            bp_ids = session.resolve_source_breakpoints(source, breakpoint_lines)
        bp1, bp2, bp3, bp4, bp5, bp6, bp7, bp8 = bp_ids

        # Expression at breakpoint 1: In main.
        stop_event = session.verify_stopped_on_breakpoint(bp1, after=cfg.process_event)
        main_frames = session.thread_context_from(stop_event).frames(levels=2)
        main_frame, caller_frame = main_frames[0], main_frames[1]
        self.set_evaluation_frame(main_frame)

        assert_eval("var1", "20", type="int")

        # In repl context, an empty expression repeats the previous expression.
        if context == "repl":
            assert_eval("", "20")
        else:
            assert_eval_fails("")

        assert_eval("var2", "21", type="int")
        if context == "repl":
            assert_eval("", "21", type="int")
            assert_eval("", "21", type="int")

        # Verify hex and decimal formatting.
        assert_eval("static_int", "0x0000002a", type="int", as_hex=True)
        assert_eval("static_int", "42", type="int")
        assert_eval("non_static_int", "0x0000002b", type="int", as_hex=True)
        assert_eval("non_static_int", "43", type="int")
        assert_eval("struct1.foo", "0x0000000f", type="int", as_hex=True)
        assert_eval("struct1.foo", "15", type="int")
        assert_eval("struct2->foo", "0x00000010", type="int", as_hex=True)
        assert_eval("struct2->foo", "16", type="int")

        if is_result_expanded:
            struct1_match = r"\(my_struct\) (struct1|\$\d+) = \(foo = 15\)"
        elif is_result_brief:
            struct1_match = r"\(foo = 15\)"
        elif is_result_summary:
            struct1_match = r"\{foo:15\}"
        else:
            struct1_match = "my_struct"
        assert_eval("struct1", struct1_match, type="my_struct", has_var_ref=True)

        if is_result_expanded:
            struct2_match = r"\(my_struct \*\) (struct2|\$\d+) = 0x.*"
        elif is_result_summary:
            struct2_match = r"0x.* \{foo:16\}"
        else:
            struct2_match = r"0x.*"
        assert_eval("struct2", struct2_match, type="my_struct *", has_var_ref=True)

        if is_result_expanded:
            struct3_match = r"\(my_struct \*\) (struct3|\$\d+) = nullptr"
        elif is_result_brief:
            struct3_match = "nullptr"
        else:
            struct3_match = r"0x.*0"
        assert_eval("struct3", struct3_match, type="my_struct *", has_var_ref=True)

        if context in ("repl", None):
            # In repl or unknown context expressions may be interpreted as lldb
            # commands since no variables have the same name as the command.
            eval_body = main_frame.evaluate("list")
            session.verify_evaluate(eval_body, matches=r".*", has_mem_ref=False)
            # Changing the frame should not make a difference.
            eval_body = caller_frame.evaluate("version")
            session.verify_evaluate(eval_body, matches=r".*lldb.+", has_mem_ref=False)
        else:
            assert_eval_fails("list")  # local variable of a_function.
            assert_eval_fails("version")

        # Identifiers and variables not in scope should fail.
        assert_eval_fails("my_struct")  # struct name.
        assert_eval_fails("int")  # type name.
        assert_eval_fails("foo")  # member variable of my_struct.

        if context_parses_expressions:
            assert_eval(
                "a_function",
                r"0x.*a.out`a_function.*",
                type="int (*)(int)",
                has_var_ref=True,
                has_mem_ref=False,
                has_loc_ref=True,
            )
            assert_eval("a_function(1)", "1", type="int", has_mem_ref=False)
            assert_eval("var2 + struct1.foo", "36", has_mem_ref=False)
            assert_eval(
                "foo_func",
                r"0x.*a.out`foo_func.*",
                type="int (*)()",
                has_var_ref=True,
                has_mem_ref=False,
                has_loc_ref=True,
            )
        else:
            assert_eval_fails("a_function")
            assert_eval_fails("a_function(1)")
            assert_eval_fails("var2 + struct1.foo")
            assert_eval_fails("foo_func")
            assert_eval_fails("(float) var2")

        # foo_var is a global variable and should evaluate.
        assert_eval("foo_var", "44")

        # Expressions at breakpoint 2: In an anonymous block.
        stop_event = session.continue_to_breakpoint(bp2)
        self.set_evaluation_frame(session.top_frame_from(stop_event))

        assert_eval("var1", "20")
        assert_eval("var2", "2")  # shadowed variable.
        assert_eval("static_int", "42")
        assert_eval("non_static_int", "10")  # shadowed variable.
        assert_eval("struct1", struct1_match, type="my_struct", has_var_ref=True)
        assert_eval("struct1.foo", "15")
        assert_eval("struct2->foo", "16")

        if context_parses_expressions:
            assert_eval(
                "a_function",
                r"0x.*a.out`a_function.*",
                type="int (*)(int)",
                has_var_ref=True,
                has_mem_ref=False,
                has_loc_ref=True,
            )
            assert_eval("a_function(1)", "1", has_mem_ref=False)
            assert_eval("var2 + struct1.foo", "17", has_mem_ref=False)
            assert_eval(
                "foo_func",
                r"0x.*a.out`foo_func.*",
                has_var_ref=True,
                has_mem_ref=False,
            )
        else:
            assert_eval_fails("a_function")
            assert_eval_fails("a_function(1)")
            assert_eval_fails("var2 + struct1.foo")
            assert_eval_fails("foo_func")
        assert_eval("foo_var", "44")

        # Expressions at breakpoint 3: In a_function.
        stop_event = session.continue_to_breakpoint(bp3)
        a_function_frames = session.thread_context_from(stop_event).frames(levels=2)
        a_frame, a_function_parent_frame = a_function_frames[0], a_function_frames[1]
        self.set_evaluation_frame(a_frame)

        assert_eval("list", "42")
        assert_eval("static_int", "42")
        assert_eval("non_static_int", "43")
        # Verify variable from a different frame.
        eval_body = a_function_parent_frame.evaluate("var1", context=context)
        session.verify_evaluate(eval_body, matches="20")

        if context_parses_expressions:
            # Access global variable without a frame
            # Run in variable mode to avoid interpreting it as a command.
            session.evaluate("`lldb-dap repl-mode variable", context="repl")

            eval_body = session.evaluate("static_int", context=context)
            session.verify_evaluate(eval_body, matches="42", type="int")

            session.evaluate("`lldb-dap repl-mode auto", context="repl")

        # In a_function's own frame these names are out of scope.
        assert_eval_fails("var1")
        assert_eval_fails("var2")
        assert_eval_fails("struct1")
        assert_eval_fails("struct1.foo")
        assert_eval_fails("struct2->foo")
        assert_eval_fails("var2 + struct1.foo")

        if context_parses_expressions:
            assert_eval(
                "a_function",
                r"0x.*a.out`a_function.*",
                has_var_ref=True,
                has_mem_ref=False,
                has_loc_ref=True,
            )
            assert_eval("a_function(1)", "1", has_mem_ref=False)
            assert_eval("list + 1", "43", has_mem_ref=False)
            assert_eval(
                "foo_func",
                r"0x.*a.out`foo_func.*",
                has_var_ref=True,
                has_mem_ref=False,
            )
        else:
            assert_eval_fails("a_function")
            assert_eval_fails("a_function(1)")
            assert_eval_fails("list + 1")
            assert_eval_fails("foo_func")
        assert_eval("foo_var", "44")

        # Expressions at breakpoints 4-7.
        # Now we check that values are updated after stepping.
        stop_event = session.continue_to_breakpoint(bp4)
        self.set_evaluation_frame(session.top_frame_from(stop_event))
        if is_result_expanded:
            my_vec_match = (
                r"\(std::vector<int>\) \$\d+ = size=2 \{\n  \[0\] = 1\n  \[1\] = 2\n\}"
            )
        elif is_result_brief:
            my_vec_match = r"size=2 \{\n  \[0\] = 1\n  \[1\] = 2\n\}"
        else:
            my_vec_match = "size=2"
        assert_eval("my_vec", my_vec_match, has_var_ref=True)

        stop_event = session.continue_to_breakpoint(bp5)
        self.set_evaluation_frame(session.top_frame_from(stop_event))
        if is_result_expanded:
            my_vec_match = r"\(std::vector<int>\) \$\d+ = size=3 \{\n  \[0\] = 1\n  \[1\] = 2\n  \[2\] = 3\n\}"
        elif is_result_brief:
            my_vec_match = r"size=3 \{\n  \[0\] = 1\n  \[1\] = 2\n  \[2\] = 3\n\}"
        else:
            my_vec_match = "size=3"
        assert_eval("my_vec", my_vec_match, has_var_ref=True)

        if is_result_expanded:
            my_map_match = r"\(std::map<int, int>\) \$\d+ = size=2 \{\n  \[0\] = \(first = 1, second = 2\)\n  \[1\] = \(first = 2, second = 3\)\n\}"
        elif is_result_brief:
            my_map_match = r"size=2 \{\n  \[0\] = \(first = 1, second = 2\)\n  \[1\] = \(first = 2, second = 3\)\n\}"
        else:
            my_map_match = "size=2"
        assert_eval("my_map", my_map_match, has_var_ref=True)

        stop_event = session.continue_to_breakpoint(bp6)
        self.set_evaluation_frame(session.top_frame_from(stop_event))
        assert_eval("my_map", "size=3", has_var_ref=True)

        if is_result_expanded:
            my_bool_match = r"\(std::vector<bool>\) \$\d+ = size=1 {\n  \[0\] = true\n}"
        elif is_result_brief:
            my_bool_match = r"size=1 \{\n  \[0\] = true\n\}"
        else:
            my_bool_match = "size=1"
        assert_eval("my_bool_vec", my_bool_match, has_var_ref=True)

        stop_event = session.continue_to_breakpoint(bp7)
        self.set_evaluation_frame(session.top_frame_from(stop_event))
        if is_result_expanded:
            my_bool_match = r"\(std::vector<bool>\) \$\d+ = size=2 \{\n  \[0\] = true\n  \[1\] = false\n\}"
        elif is_result_brief:
            my_bool_match = r"size=2 \{\n  \[0\] = true\n  \[1\] = false\n\}"
        else:
            my_bool_match = "size=2"
        assert_eval("my_bool_vec", my_bool_match, has_var_ref=True)

        # Expressions at breakpoint 8.
        stop_event = session.continue_to_breakpoint(bp8)
        self.set_evaluation_frame(session.top_frame_from(stop_event))

        if context == "repl":
            # In repl, empty expressions repeat the previous lldb command,
            # so each call reads the next byte of my_ints.
            assert_eval("memory read -c 1 &my_ints", r".* 05 .*\n", has_mem_ref=False)
            assert_eval("", r".* 0a .*\n", has_mem_ref=False)
            assert_eval("", r".* 0f .*\n", has_mem_ref=False)
            assert_eval("", r".* 14 .*\n", has_mem_ref=False)
            assert_eval("", r".* 19 .*\n", has_mem_ref=False)

        if is_result_expanded:
            my_longs_match = (
                r"\(long\[3\]\) \$\d+ = \(\[0\] = 5, \[1\] = 6, \[2\] = 7\)"
            )
        elif is_result_brief:
            my_longs_match = r"\(\[0\] = 5, \[1\] = 6, \[2\] = 7\)"
        elif is_result_summary:
            my_longs_match = r"\{5, 6, 7\}"
        else:
            my_longs_match = r"long\[3\]"
        assert_eval("my_longs", my_longs_match, has_var_ref=True)

        session.continue_to_exit()

    @skipIfWindows
    def test_generic_evaluate_expressions(self):
        # Tests context-less expression evaluations.
        self.run_evaluate_expressions(enableAutoVariableSummaries=False)

    @skipIfWindows
    def test_repl_evaluate_expressions(self):
        # Tests expression evaluations that are triggered from the Debug Console.
        self.run_evaluate_expressions("repl", enableAutoVariableSummaries=False)

    @skipIfWindows
    def test_watch_evaluate_expressions(self):
        # Tests expression evaluations that are triggered from a watch expression.
        self.run_evaluate_expressions("watch", enableAutoVariableSummaries=True)

    @skipIfWindows
    def test_hover_evaluate_expressions(self):
        # Tests expression evaluations that are triggered when hovering on the editor.
        self.run_evaluate_expressions("hover", enableAutoVariableSummaries=False)

    @skipIfWindows
    def test_variable_evaluate_expressions(self):
        # Tests expression evaluations that are triggered in the variable explorer.
        self.run_evaluate_expressions("variables", enableAutoVariableSummaries=True)

    @skipIfWindows
    def test_clipboard_evaluate_expressions(self):
        # Tests expression evaluations that are triggered when value copied in editor.
        self.run_evaluate_expressions("clipboard", enableAutoVariableSummaries=False)
