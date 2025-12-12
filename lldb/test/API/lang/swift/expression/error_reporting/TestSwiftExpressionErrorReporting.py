import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftExpressionErrorReporting(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test_missing_location(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        process.Continue()
        self.ci.HandleCommand(
            "settings set symbols.testing.inject-variable-location-error true", self.res
        )
        if not self.res.Succeeded():
            # This test needs assertions.
            return

        options = lldb.SBExpressionOptions()
        value = self.frame().EvaluateExpression("number", options)
        data = value.GetError().GetErrorData()
        version = data.GetValueForKey("version")
        self.assertEqual(version.GetIntegerValue(), 1)
        diags = data.GetValueForKey("errors").GetItemAtIndex(0)
        details = diags.GetValueForKey("details")
        all_messages = [str(detail.GetValueForKey("message")) for detail in details]
        self.assertIn(
            'Missing debug information for variable "self": variable not available',
            all_messages,
        )

    @swiftTest
    def test_missing_var(self):
        """Test error reporting in expressions reports
        only diagnostics in user code"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # This produces two errors:
        #   error: <EXPR>:8:1: initializers may only be declared within a type
        #   } catch (let __lldb_tmp_error) {
        #   ^
        #
        #   error: <EXPR>:6:5: expected '(' for initializer parameters
        #   init
        #       ^
        # The first one is outside of user code, the second one isn't.

        options = lldb.SBExpressionOptions()
        value = self.frame().EvaluateExpression("init", options)
        data = value.GetError().GetErrorData()

        version = data.GetValueForKey("version")
        self.assertEqual(version.GetIntegerValue(), 1)
        diags = data.GetValueForKey("errors").GetItemAtIndex(0)
        details = diags.GetValueForKey("details")
        err0 = details.GetItemAtIndex(0)
        self.assertIn("initializers", str(err0.GetValueForKey("message")))
        loc0 = err0.GetValueForKey("source_location")
        self.assertTrue(loc0.GetValueForKey("hidden").GetBooleanValue())
        self.assertFalse(loc0.GetValueForKey("in_user_input").GetBooleanValue())
        err1 = details.GetItemAtIndex(1)
        self.assertIn("expected '('", str(err1.GetValueForKey("message")))
        loc1 = err1.GetValueForKey("source_location")
        self.assertFalse(loc1.GetValueForKey("hidden").GetBooleanValue())
        self.assertTrue(loc1.GetValueForKey("in_user_input").GetBooleanValue())

        options = lldb.SBExpressionOptions()
        value = self.frame().EvaluateExpression(
            "ceciNestPasUnVar", options)
        def check(value):
            lines = str(value.GetError()).split('\n')
            self.assertTrue(lines[0].startswith('error:'))
            self.assertIn('ceciNestPasUnVar', lines[0])
            for line in lines[1:]:
                self.assertFalse(line.startswith('error:'))
                self.assertFalse(line.startswith('warning:'))

        check(value)
        process.Continue()
        value = self.frame().EvaluateExpression(
            "ceciNestPasUnVar", options)
        check(value)

    @swiftTest
    def test_missing_type(self):
        """Test error reporting in expressions reports
        only diagnostics in user code"""
        self.build(dictionary={'HIDE_SWIFTMODULE': 'YES'})
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        options = lldb.SBExpressionOptions()
        value = self.frame().EvaluateExpression("strct", options)
        def check(value):
            lines = str(value.GetError()).split('\n')
            self.assertTrue(lines[0].startswith('error:'))
            self.assertIn('Missing debug info', lines[0])
            self.assertIn('strct', lines[0])

        check(value)

        # This succeeds using stringForPrintObject(_:mangledTypeName:), which
        # doesn't require the type to be available.
        # Note: (?s)^(?!.*<pattern>) checks that the pattern is not found.
        self.expect(
            "dwim-print -O -- strct",
            patterns=["(?s)^(?!.*error: Missing type)", "properties : true"],
        )

        process.Continue()
        self.expect('expression -O -- number', error=True,
                    substrs=['self', 'not', 'found'])
