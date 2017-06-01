# RUN: %{python} %s
#
# END.


import unittest
import platform
import os.path
import tempfile

import lit
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
            IntegratedTestKeywordParser("MY_RUN:", ParserKind.COMMAND),
            IntegratedTestKeywordParser("MY_CUSTOM:", ParserKind.CUSTOM,
                                        custom_parse)
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
        self.assertItemsEqual(list_parser.getValue(),
                              ['one', 'two', 'three', 'four'])

    def test_commands(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        cmd_parser = self.get_parser(parsers, 'MY_RUN:')
        value = cmd_parser.getValue()
        self.assertEqual(len(value), 2)  # there are only two run lines
        self.assertEqual(value[0].strip(), 'baz')
        self.assertEqual(value[1].strip(), 'foo  bar')

    def test_custom(self):
        parsers = self.make_parsers()
        self.parse_test(parsers)
        custom_parser = self.get_parser(parsers, 'MY_CUSTOM:')
        value = custom_parser.getValue()
        self.assertItemsEqual(value, ['a', 'b', 'c'])


if __name__ == '__main__':
    TestIntegratedTestKeywordParser.load_keyword_parser_lit_tests()
    unittest.main(verbosity=2)
