import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class OdrHandlingWithDylibTestCase(TestBase):
    @skipIf(
        bugnumber="https://github.com/llvm/llvm-project/issues/50375, rdar://135551810"
    )
    def test(self):
        """
        Tests that the expression evaluator is able to deal with types
        whose definitions conflict across multiple LLDB modules (in this
        case the definition for 'class Service' in the main executable
        has an additional field compared to the definition found in the
        dylib). This causes the ASTImporter to detect a name conflict
        while importing 'Service'. With LLDB's liberal ODRHandlingType
        the ASTImporter happily creates a conflicting AST node for
        'Service' in the scratch ASTContext, leading to a crash down
        the line.
        """
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "plugin_entry", lldb.SBFileSpec("plugin.cpp")
        )

        self.expect_expr("*gProxyThis")
