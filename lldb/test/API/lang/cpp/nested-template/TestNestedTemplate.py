"""
Test that a nested template parameter works with simple template names.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class NestedTemplateTestCase(TestBase):
    def do_test(self, debug_flags):
        self.build(dictionary=debug_flags)
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.expect(
            "image lookup -A -t 'Inner<int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'NS::Struct<int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'NS::Union<int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_simple_template_names(self):
        self.do_test(dict(TEST_CFLAGS_EXTRAS="-gsimple-template-names"))

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_no_simple_template_names(self):
        self.do_test(dict(TEST_CFLAGS_EXTRAS="-gno-simple-template-names"))
