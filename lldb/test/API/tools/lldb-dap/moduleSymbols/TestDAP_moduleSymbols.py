"""
Test lldb-dap moduleSymbols request
"""

import lldbdap_testcase


class TestDAP_moduleSymbols(lldbdap_testcase.DAPTestCaseBase):
    def test_moduleSymbols(self):
        """
        Test that the moduleSymbols request returns correct symbols from the module.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        symbol_names = []
        i = 0
        while True:
            next_symbol = self.dap_server.request_moduleSymbols(
                moduleName="a.out", startIndex=i, count=1
            )
            self.assertIn("symbols", next_symbol["body"])
            result_symbols = next_symbol["body"]["symbols"]
            self.assertLessEqual(len(result_symbols), 1)
            if len(result_symbols) == 0:
                break

            self.assertIn("name", result_symbols[0])
            symbol_names.append(result_symbols[0]["name"])
            i += 1
            if i >= 1000:
                break

        self.assertGreater(len(symbol_names), 0)
        self.assertIn("main", symbol_names)
        self.assertIn("func1", symbol_names)
        self.assertIn("func2", symbol_names)
