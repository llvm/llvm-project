from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        if self.expectedCompiler(["clang"]) and self.expectedCompilerVersion(['>', '16.0']):
            vec_type = "std::vector<int>"
        else:
            vec_type = "std::vector<int, std::allocator<int> >"

        # Test printing the vector before enabling any C++ module setting.
        self.expect_expr("a", result_type=vec_type)

        # Set loading the import-std-module to 'fallback' which loads the module
        # and retries when an expression fails to parse.
        self.runCmd("settings set target.import-std-module fallback")

        # Printing the vector still works. This should return the same type
        # as before as this shouldn't use a C++ module type.
        self.expect_expr("a", result_type=vec_type)

        # This expression can only parse with a C++ module. LLDB should
        # automatically fall back to import the C++ module to get this working.
        self.expect_expr("std::max<std::size_t>(0U, a.size())", result_value="3")


        # The 'a' and 'local' part can be parsed without loading a C++ module and will
        # load type/runtime information. The 'std::max...' part will fail to
        # parse without a C++ module. Make sure we reset all the relevant parts of
        # the C++ parser so that we don't end up with for example a second
        # definition of 'local' when retrying.
        self.expect_expr("a; local; std::max<std::size_t>(0U, a.size())", result_value="3")


        # Try to declare top-level declarations that require a C++ module to parse.
        # Top-level expressions don't support importing the C++ module (yet), so
        # this should still fail as before.
        self.expect("expr --top-level -- int i = std::max(1, 2);", error=True,
                    substrs=["no member named 'max' in namespace 'std'"])

        # The proper diagnostic however should be shown on the retry.
        self.expect("expr std::max(1, 2); unknown_identifier", error=True,
                    substrs=["use of undeclared identifier 'unknown_identifier'"])

        # Turn on the 'import-std-module' setting and make sure we import the
        # C++ module.
        self.runCmd("settings set target.import-std-module true")
        # This is still expected to work.
        self.expect_expr("std::max<std::size_t>(0U, a.size())", result_value="3")

        # Turn of the 'import-std-module' setting and make sure we don't load
        # the module (which should prevent parsing the expression involving
        # 'std::max').
        self.runCmd("settings set target.import-std-module false")
        self.expect("expr std::max(1, 2);", error=True,
                    substrs=["no member named 'max' in namespace 'std'"])
