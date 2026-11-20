# RUN: %{python} %s
#
# END.

import io
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import lit.Test as Test
from lit.reports import XunitReport


class TestXunitReport(unittest.TestCase):
    def render(self, code, output, requires=()):
        config = SimpleNamespace(name="suite", available_features=set())
        suite = Test.TestSuite("suite", "", "", config)
        test = Test.Test(suite, ("test",), config)
        test.requires = list(requires)
        test.result = Test.Result(code, output, elapsed=0.5)
        stream = io.StringIO()
        XunitReport(None)._write_results_to_file([test], 0.5, stream)
        return ET.fromstring(stream.getvalue()).find("testsuite")

    def test_skipped_output(self):
        output = 'UNSUPPORTED: test (requires <Darwin> & "debug") ]]>\n\x1b'
        for code in (Test.SKIPPED, Test.UNSUPPORTED):
            with self.subTest(code=code):
                suite = self.render(code, output)
                self.assertEqual(suite.get("tests"), "1")
                self.assertEqual(suite.get("skipped"), "1")
                self.assertEqual(suite.get("failures"), "0")
                case = suite.find("testcase")
                self.assertEqual(case.findtext("system-out"), output[:-1])
                expected = (
                    "Skipped" if code == Test.SKIPPED else "Unsupported configuration"
                )
                self.assertEqual(case.find("skipped").get("message"), expected)

    def test_empty_skipped_output(self):
        for code in (Test.SKIPPED, Test.UNSUPPORTED):
            with self.subTest(code=code):
                case = self.render(code, "").find("testcase")
                self.assertIsNotNone(case.find("skipped"))
                self.assertIsNone(case.find("system-out"))

    def test_missing_feature(self):
        case = self.render(Test.UNSUPPORTED, "requires Darwin", ["Darwin"]).find(
            "testcase"
        )
        self.assertEqual(
            case.find("skipped").get("message"), "Missing required feature(s): Darwin"
        )
        self.assertEqual(case.findtext("system-out"), "requires Darwin")

    def test_excluded_output(self):
        case = self.render(Test.EXCLUDED, "not run").find("testcase")
        self.assertEqual(
            case.find("skipped").get("message"),
            "Test not selected (--filter, --max-tests)",
        )
        self.assertIsNone(case.find("system-out"))

    def test_passing_output(self):
        for code in (Test.PASS, Test.XFAIL):
            with self.subTest(code=code):
                case = self.render(code, "passing output").find("testcase")
                self.assertEqual(len(case), 0)

    def test_failure_output(self):
        output = "failure < & ]]>\n\x1b"
        suite = self.render(Test.FAIL, output)
        self.assertEqual(suite.get("failures"), "1")
        self.assertEqual(suite.get("skipped"), "0")
        case = suite.find("testcase")
        self.assertEqual(case.findtext("failure"), output[:-1])
        self.assertIsNone(case.find("system-out"))


if __name__ == "__main__":
    unittest.main()
