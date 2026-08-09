# RUN: %{python} %s
#
# END.


import os
import os.path
import platform
import shutil
import tempfile
import unittest

import lit.discovery
import lit.LitConfig
import lit.Test as Test
import lit.TestRunner
from lit.ShellEnvironment import ShellEnvironment
from lit.ShUtil import ShParser
from lit.TestRunner import (
    ParserKind,
    IntegratedTestKeywordParser,
    executeShCmd,
    parseIntegratedTestScript,
)


class TestIntegratedTestKeywordParser(unittest.TestCase):
    inputTestCase = None

    @staticmethod
    def load_keyword_parser_lit_tests():
        """
        Create and load the LIT test suite and test objects used by
        TestIntegratedTestKeywordParser
        """
        # Create the global config object.
        lit_config = lit.LitConfig.LitConfig(
            progname="lit",
            path=[],
            diagnostic_level="note",
            useValgrind=False,
            valgrindLeakCheck=False,
            valgrindArgs=[],
            noExecute=False,
            debug=False,
            isWindows=(platform.system() == "Windows"),
            order="smart",
            params={},
        )
        TestIntegratedTestKeywordParser.litConfig = lit_config
        # Perform test discovery.
        test_path = os.path.dirname(os.path.dirname(__file__))
        inputs = [os.path.join(test_path, "Inputs/testrunner-custom-parsers/")]
        assert os.path.isdir(inputs[0])
        tests = lit.discovery.find_tests_for_inputs(lit_config, inputs)
        assert len(tests) == 1 and "there should only be one test"
        TestIntegratedTestKeywordParser.inputTestCase = tests[0]

    @staticmethod
    def make_parsers():
        def custom_parse(line_number, line, output):
            if output is None:
                output = []
            output += [part for part in line.split(" ") if part.strip()]
            return output

        return [
            IntegratedTestKeywordParser("MY_TAG.", ParserKind.TAG),
            IntegratedTestKeywordParser("MY_DNE_TAG.", ParserKind.TAG),
            IntegratedTestKeywordParser("MY_LIST:", ParserKind.LIST),
            IntegratedTestKeywordParser("MY_SPACE_LIST:", ParserKind.SPACE_LIST),
            IntegratedTestKeywordParser("MY_BOOL:", ParserKind.BOOLEAN_EXPR),
            IntegratedTestKeywordParser("MY_INT:", ParserKind.INTEGER),
            IntegratedTestKeywordParser("MY_RUN:", ParserKind.COMMAND),
            IntegratedTestKeywordParser("MY_CUSTOM:", ParserKind.CUSTOM, custom_parse),
            IntegratedTestKeywordParser("MY_DEFINE:", ParserKind.DEFINE),
            IntegratedTestKeywordParser("MY_REDEFINE:", ParserKind.REDEFINE),
        ]

    @staticmethod
    def get_parser(parser_list, keyword):
        for p in parser_list:
            if p.keyword == keyword:
                return p
        assert False and "parser not found"

    @staticmethod
    def parse_test(parser_list, allow_result=False):
        script = parseIntegratedTestScript(
            TestIntegratedTestKeywordParser.inputTestCase,
            additional_parsers=parser_list,
            require_script=False,
        )
        if isinstance(script, lit.Test.Result):
            assert allow_result
        else:
            assert isinstance(script, list)
            assert len(script) == 0
        return script

    def test_tags(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        tag_parser = self.get_parser(parsers, "MY_TAG.")
        dne_tag_parser = self.get_parser(parsers, "MY_DNE_TAG.")
        self.assertTrue(tag_parser.getValue())
        self.assertFalse(dne_tag_parser.getValue())

    def test_lists(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        list_parser = self.get_parser(parsers, "MY_LIST:")
        self.assertEqual(list_parser.getValue(), ["one", "two", "three", "four"])

    def test_space_lists(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        space_list_parser = self.get_parser(parsers, "MY_SPACE_LIST:")
        self.assertEqual(
            space_list_parser.getValue(),
            [
                "orange",
                "tabby",
                "tortie",
                "tuxedo",
                "void",
                "multiple",
                "spaces",
                "cute,",
                "fluffy,",
                "kittens",
            ],
        )

    def test_commands(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        cmd_parser = self.get_parser(parsers, "MY_RUN:")
        value = cmd_parser.getValue()
        self.assertEqual(len(value), 2)  # there are only two run lines
        self.assertEqual(value[0].command.strip(), "%dbg(MY_RUN: at line 4)  baz")
        self.assertEqual(value[1].command.strip(), "%dbg(MY_RUN: at line 12)  foo  bar")

    def test_boolean(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        bool_parser = self.get_parser(parsers, "MY_BOOL:")
        value = bool_parser.getValue()
        self.assertEqual(len(value), 2)  # there are only two run lines
        self.assertEqual(value[0].strip(), "a && (b)")
        self.assertEqual(value[1].strip(), "d")

    def test_integer(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        int_parser = self.get_parser(parsers, "MY_INT:")
        value = int_parser.getValue()
        self.assertEqual(len(value), 2)  # there are only two MY_INT: lines
        self.assertEqual(type(value[0]), int)
        self.assertEqual(value[0], 4)
        self.assertEqual(type(value[1]), int)
        self.assertEqual(value[1], 6)

    def test_bad_parser_type(self):
        parsers = self.make_parsers() + ["BAD_PARSER_TYPE"]
        script = self.parse_test(parsers, allow_result=True)
        self.assertTrue(isinstance(script, lit.Test.Result))
        self.assertEqual(script.code, lit.Test.UNRESOLVED)
        self.assertEqual(
            "Additional parser must be an instance of " "IntegratedTestKeywordParser",
            script.output,
        )

    def test_duplicate_keyword(self):
        parsers = self.make_parsers() + [
            IntegratedTestKeywordParser("KEY:", ParserKind.BOOLEAN_EXPR),
            IntegratedTestKeywordParser("KEY:", ParserKind.BOOLEAN_EXPR),
        ]
        script = self.parse_test(parsers, allow_result=True)
        self.assertTrue(isinstance(script, lit.Test.Result))
        self.assertEqual(script.code, lit.Test.UNRESOLVED)
        self.assertEqual("Parser for keyword 'KEY:' already exists", script.output)

    def test_boolean_unterminated(self):
        parsers = self.make_parsers() + [
            IntegratedTestKeywordParser(
                "MY_BOOL_UNTERMINATED:", ParserKind.BOOLEAN_EXPR
            )
        ]
        script = self.parse_test(parsers, allow_result=True)
        self.assertTrue(isinstance(script, lit.Test.Result))
        self.assertEqual(script.code, lit.Test.UNRESOLVED)
        self.assertEqual(
            "Test has unterminated 'MY_BOOL_UNTERMINATED:' lines " "(with '\\')",
            script.output,
        )

    def test_custom(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        custom_parser = self.get_parser(parsers, "MY_CUSTOM:")
        value = custom_parser.getValue()
        self.assertEqual(value, ["a", "b", "c"])

    def test_defines(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        cmd_parser = self.get_parser(parsers, "MY_DEFINE:")
        value = cmd_parser.getValue()
        self.assertEqual(len(value), 1)  # there's only one MY_DEFINE directive
        self.assertEqual(value[0].new_subst, True)
        self.assertEqual(value[0].name, "%{name}")
        self.assertEqual(value[0].value, "value one")

    def test_redefines(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        cmd_parser = self.get_parser(parsers, "MY_REDEFINE:")
        value = cmd_parser.getValue()
        self.assertEqual(len(value), 1)  # there's only one MY_REDEFINE directive
        self.assertEqual(value[0].new_subst, False)
        self.assertEqual(value[0].name, "%{name}")
        self.assertEqual(value[0].value, "value two")

    def test_bad_keywords(self):
        def custom_parse(line_number, line, output):
            return output

        try:
            IntegratedTestKeywordParser("TAG_NO_SUFFIX", ParserKind.TAG),
            self.fail("TAG_NO_SUFFIX failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("TAG_NO_SUFFIX raised the wrong exception: %r" % e)

        try:
            IntegratedTestKeywordParser("TAG_WITH_COLON:", ParserKind.TAG),
            self.fail("TAG_WITH_COLON: failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("TAG_WITH_COLON: raised the wrong exception: %r" % e)

        try:
            IntegratedTestKeywordParser("LIST_WITH_DOT.", ParserKind.LIST),
            self.fail("LIST_WITH_DOT. failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("LIST_WITH_DOT. raised the wrong exception: %r" % e)

        try:
            IntegratedTestKeywordParser("SPACE_LIST_WITH_DOT.", ParserKind.SPACE_LIST),
            self.fail("SPACE_LIST_WITH_DOT. failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("SPACE_LIST_WITH_DOT. raised the wrong exception: %r" % e)

        try:
            IntegratedTestKeywordParser(
                "CUSTOM_NO_SUFFIX", ParserKind.CUSTOM, custom_parse
            ),
            self.fail("CUSTOM_NO_SUFFIX failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("CUSTOM_NO_SUFFIX raised the wrong exception: %r" % e)

        # Both '.' and ':' are allowed for CUSTOM keywords.
        try:
            IntegratedTestKeywordParser(
                "CUSTOM_WITH_DOT.", ParserKind.CUSTOM, custom_parse
            ),
        except BaseException as e:
            self.fail("CUSTOM_WITH_DOT. raised an exception: %r" % e)
        try:
            IntegratedTestKeywordParser(
                "CUSTOM_WITH_COLON:", ParserKind.CUSTOM, custom_parse
            ),
        except BaseException as e:
            self.fail("CUSTOM_WITH_COLON: raised an exception: %r" % e)

        try:
            IntegratedTestKeywordParser("CUSTOM_NO_PARSER:", ParserKind.CUSTOM),
            self.fail("CUSTOM_NO_PARSER: failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("CUSTOM_NO_PARSER: raised the wrong exception: %r" % e)


class TestApplySubtitutions(unittest.TestCase):
    def test_simple(self):
        script = ["echo %bar"]
        substitutions = [("%bar", "hello")]
        result = lit.TestRunner.applySubstitutions(script, substitutions)
        self.assertEqual(result, ["echo hello"])

    def test_multiple_substitutions(self):
        script = ["echo %bar %baz"]
        substitutions = [
            ("%bar", "hello"),
            ("%baz", "world"),
            ("%useless", "shouldnt expand"),
        ]
        result = lit.TestRunner.applySubstitutions(script, substitutions)
        self.assertEqual(result, ["echo hello world"])

    def test_multiple_script_lines(self):
        script = ["%cxx %compile_flags -c -o %t.o", "%cxx %link_flags %t.o -o %t.exe"]
        substitutions = [
            ("%cxx", "clang++"),
            ("%compile_flags", "-std=c++11 -O3"),
            ("%link_flags", "-lc++"),
        ]
        result = lit.TestRunner.applySubstitutions(script, substitutions)
        self.assertEqual(
            result,
            ["clang++ -std=c++11 -O3 -c -o %t.o", "clang++ -lc++ %t.o -o %t.exe"],
        )

    def test_recursive_substitution_real(self):
        script = ["%build %s"]
        substitutions = [
            ("%cxx", "clang++"),
            ("%compile_flags", "-std=c++11 -O3"),
            ("%link_flags", "-lc++"),
            ("%build", "%cxx %compile_flags %link_flags %s -o %t.exe"),
        ]
        result = lit.TestRunner.applySubstitutions(
            script, substitutions, recursion_limit=3
        )
        self.assertEqual(result, ["clang++ -std=c++11 -O3 -lc++ %s -o %t.exe %s"])

    def test_recursive_substitution_limit(self):
        script = ["%rec5"]
        # Make sure the substitutions are not in an order where the global
        # substitution would appear to be recursive just because they are
        # processed in the right order.
        substitutions = [
            ("%rec1", "STOP"),
            ("%rec2", "%rec1"),
            ("%rec3", "%rec2"),
            ("%rec4", "%rec3"),
            ("%rec5", "%rec4"),
        ]
        for limit in [5, 6, 7]:
            result = lit.TestRunner.applySubstitutions(
                script, substitutions, recursion_limit=limit
            )
            self.assertEqual(result, ["STOP"])

    def test_recursive_substitution_limit_exceeded(self):
        script = ["%rec5"]
        substitutions = [
            ("%rec1", "STOP"),
            ("%rec2", "%rec1"),
            ("%rec3", "%rec2"),
            ("%rec4", "%rec3"),
            ("%rec5", "%rec4"),
        ]
        for limit in [0, 1, 2, 3, 4]:
            try:
                lit.TestRunner.applySubstitutions(
                    script, substitutions, recursion_limit=limit
                )
                self.fail("applySubstitutions should have raised an exception")
            except ValueError:
                pass

    def test_recursive_substitution_invalid_value(self):
        script = ["%rec5"]
        substitutions = [
            ("%rec1", "STOP"),
            ("%rec2", "%rec1"),
            ("%rec3", "%rec2"),
            ("%rec4", "%rec3"),
            ("%rec5", "%rec4"),
        ]
        for limit in [-1, -2, -3, "foo"]:
            try:
                lit.TestRunner.applySubstitutions(
                    script, substitutions, recursion_limit=limit
                )
                self.fail("applySubstitutions should have raised an exception")
            except AssertionError:
                pass


class TestBuiltinSpawnFallbackPythonPath(unittest.TestCase):
    """Regression test for https://github.com/llvm/llvm-project/issues/191131:

    The spawn-fallback path used for builtin 'cat'/'diff' commands (taken
    when the command can't run in-process, e.g. because it's wrapped in
    'not --crash') must not permanently mutate the shared
    ShellEnvironment's PYTHONPATH, or it leaks into every later RUN line
    that shares the same shell environment.

    Forcing this path normally requires the 'not --crash' wrapper, which
    in turn requires the real 'not' helper binary that ships with a full
    LLVM/Clang build -- unavailable in a pure-Python test environment.
    Instead, this test drives the real executeShCmd()/_executeShCmd() path
    for a plain 'cat' command (so cmd_shenv is the pipeline's own shenv,
    exactly as it is for 'not --crash') and monkeypatches only the
    use-in-process decision (lit.TestRunner._should_run_inproc) to return
    False, so the real spawn-fallback code -- the code under test -- still
    runs unmodified and unmocked.
    """

    def test_cat_spawn_fallback_does_not_leak_pythonpath(self):
        tmpdir = tempfile.mkdtemp()
        try:
            input_path = os.path.join(tmpdir, "input.txt")
            with open(input_path, "w") as f:
                f.write("hello\n")

            env = dict(os.environ)
            sentinel = "sentinel-pythonpath-value"
            env["PYTHONPATH"] = sentinel
            shenv = ShellEnvironment(tmpdir, env)

            # Shell-quote the path so ShParser doesn't treat any backslashes
            # (e.g. on Windows) as escape characters.
            cmd = ShParser('cat "%s"' % input_path.replace("\\", "\\\\")).parse()

            original_should_run_inproc = lit.TestRunner._should_run_inproc
            lit.TestRunner._should_run_inproc = lambda *args, **kwargs: False
            try:
                results = []
                exitCode, timeoutInfo = executeShCmd(cmd, shenv, results)
            finally:
                lit.TestRunner._should_run_inproc = original_should_run_inproc

            self.assertEqual(timeoutInfo, None)
            self.assertEqual(exitCode, 0)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].stdout, "hello\n")

            # The bug: the spawn-fallback path used to mutate
            # cmd_shenv.env["PYTHONPATH"] in place, and here cmd_shenv is
            # shenv itself (no 'env'/'not' prefix forked a private copy),
            # so that would corrupt PYTHONPATH for every subsequent RUN
            # line sharing this ShellEnvironment.
            self.assertEqual(shenv.env["PYTHONPATH"], sentinel)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    TestIntegratedTestKeywordParser.load_keyword_parser_lit_tests()
    unittest.main(verbosity=2)
