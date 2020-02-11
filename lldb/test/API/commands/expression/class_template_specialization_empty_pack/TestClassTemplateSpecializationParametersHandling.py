"""
Test Expression Parser code gen for ClassTemplateSpecializationDecl to insure
that we generate a TemplateTypeParmDecl in the TemplateParameterList for empty
variadic packs.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestClassTemplateSpecializationParametersHandling(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_class_template_specialization(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.cpp", False))
        self.expect_expr("b.foo()", result_type="int", result_value="1")
