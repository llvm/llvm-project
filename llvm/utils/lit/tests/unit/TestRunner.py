# RUN: %{python} %s
#
# END.


import unittest
import platform
import os.path
import tempfile

import lit
import lit.Test as Test
from lit.TestRunner import ParserKind, IntegratedTestKeywordParser, \
                           parseIntegratedTestScript


class TestIntegratedTestKeywordParser(unittest.TestCase):
    inputTestCase = None

    @staticmethod
    def load_keyword_parser_lit_tests():
        """
        Create and load the LIT test suite and test objects used by
        TestIntegratedTestKeywordParser
        """
        # Create the global config object.
        lit_config = lit.LitConfig.LitConfig(progname='lit',
                                             path=[],
                                             quiet=False,
                                             useValgrind=False,
                                             valgrindLeakCheck=False,
                                             singleProcess=False,
                                             valgrindArgs=[],
                                             noExecute=False,
                                             debug=False,
                                             isWindows=(
                                               platform.system() == 'Windows'),
                                             params={})
        TestIntegratedTestKeywordParser.litConfig = lit_config
        # Perform test discovery.
        test_path = os.path.dirname(os.path.dirname(__file__))
        inputs = [os.path.join(test_path, 'Inputs/testrunner-custom-parsers/')]
        assert os.path.isdir(inputs[0])
        run = lit.run.Run(lit_config,
                          lit.discovery.find_tests_for_inputs(lit_config, inputs))
        assert len(run.tests) == 1 and "there should only be one test"
        TestIntegratedTestKeywordParser.inputTestCase = run.tests[0]

    @staticmethod
    def make_parsers():
        def custom_parse(line_number, line, output):
            if output is None:
                output = []
            output += [part for part in line.split(' ') if part.strip()]
            return output

        return [
            IntegratedTestKeywordParser("MY_TAG.", ParserKind.TAG),
            IntegratedTestKeywordParser("MY_DNE_TAG.", ParserKind.TAG),
            IntegratedTestKeywordParser("MY_LIST:", ParserKind.LIST),
            IntegratedTestKeywordParser("MY_BOOL:", ParserKind.BOOLEAN_EXPR),
            IntegratedTestKeywordParser("MY_RUN:", ParserKind.COMMAND),
            IntegratedTestKeywordParser("MY_CUSTOM:", ParserKind.CUSTOM,
                                        custom_parse),

        ]

    @staticmethod
    def get_parser(parser_list, keyword):
        for p in parser_list:
            if p.keyword == keyword:
                return p
        assert False and "parser not found"

    @staticmethod
    def parse_test(parser_list):
        script = parseIntegratedTestScript(
            TestIntegratedTestKeywordParser.inputTestCase,
            additional_parsers=parser_list, require_script=False)
        assert not isinstance(script, lit.Test.Result)
        assert isinstance(script, list)
        assert len(script) == 0

    def test_tags(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        tag_parser = self.get_parser(parsers, 'MY_TAG.')
        dne_tag_parser = self.get_parser(parsers, 'MY_DNE_TAG.')
        self.assertTrue(tag_parser.getValue())
        self.assertFalse(dne_tag_parser.getValue())

    def test_lists(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        list_parser = self.get_parser(parsers, 'MY_LIST:')
        self.assertEqual(list_parser.getValue(),
                              ['one', 'two', 'three', 'four'])

    def test_commands(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        cmd_parser = self.get_parser(parsers, 'MY_RUN:')
        value = cmd_parser.getValue()
        self.assertEqual(len(value), 2)  # there are only two run lines
        self.assertEqual(value[0].strip(), "%dbg(MY_RUN: at line 4)  baz")
        self.assertEqual(value[1].strip(), "%dbg(MY_RUN: at line 7)  foo  bar")

    def test_boolean(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        bool_parser = self.get_parser(parsers, 'MY_BOOL:')
        value = bool_parser.getValue()
        self.assertEqual(len(value), 2)  # there are only two run lines
        self.assertEqual(value[0].strip(), "a && (b)")
        self.assertEqual(value[1].strip(), "d")

    def test_boolean_unterminated(self):
        parsers = self.make_parsers() + \
            [IntegratedTestKeywordParser("MY_BOOL_UNTERMINATED:", ParserKind.BOOLEAN_EXPR)]
        try:
            self.parse_test(parsers)
            self.fail('expected exception')
        except ValueError as e:
            self.assertIn("Test has unterminated MY_BOOL_UNTERMINATED: lines", str(e))


    def test_custom(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        custom_parser = self.get_parser(parsers, 'MY_CUSTOM:')
        value = custom_parser.getValue()
        self.assertEqual(value, ['a', 'b', 'c'])

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
            IntegratedTestKeywordParser("CUSTOM_NO_SUFFIX",
                                        ParserKind.CUSTOM, custom_parse),
            self.fail("CUSTOM_NO_SUFFIX failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("CUSTOM_NO_SUFFIX raised the wrong exception: %r" % e)

        # Both '.' and ':' are allowed for CUSTOM keywords.
        try:
            IntegratedTestKeywordParser("CUSTOM_WITH_DOT.",
                                        ParserKind.CUSTOM, custom_parse),
        except BaseException as e:
            self.fail("CUSTOM_WITH_DOT. raised an exception: %r" % e)
        try:
            IntegratedTestKeywordParser("CUSTOM_WITH_COLON:",
                                        ParserKind.CUSTOM, custom_parse),
        except BaseException as e:
            self.fail("CUSTOM_WITH_COLON: raised an exception: %r" % e)

        try:
            IntegratedTestKeywordParser("CUSTOM_NO_PARSER:",
                                        ParserKind.CUSTOM),
            self.fail("CUSTOM_NO_PARSER: failed to raise an exception")
        except ValueError as e:
            pass
        except BaseException as e:
            self.fail("CUSTOM_NO_PARSER: raised the wrong exception: %r" % e)

if __name__ == '__main__':
    TestIntegratedTestKeywordParser.load_keyword_parser_lit_tests()
    unittest.main(verbosity=2)
