from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class GenericUnorderedDataFormatterTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def setUp(self):
        TestBase.setUp(self)
        self.namespace = "std"

    def do_test_with_run_command(self):
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(self, "Set break point at this line.")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)
            self.runCmd("settings set auto-one-line-summaries true", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        ns = self.namespace

        # We check here that the map shows 0 children even with corrupt data.
        self.look_for_content_and_continue(
            "corrupt_map", ["%s::unordered_map" % ns, "size=0 {}"]
        )

        # Ensure key/value children, not wrapped in a layer.
        # This regex depends on auto-one-line-summaries.
        self.runCmd("settings set auto-one-line-summaries false")
        children_are_key_value = r"\[0\] = \{\s*first = "

        unordered_map_type = (
            "std::unordered_map<int, std::basic_string<char, std::char_traits<char>, std::allocator<char>>, std::hash<int>, std::equal_to<int>, std::allocator<std::pair<int const, std::basic_string<char, std::char_traits<char>, std::allocator<char>>>>>"
            if self.getDebugInfo() == "pdb"
            else "UnorderedMap"
        )
        self.look_for_content_and_continue(
            "map",
            [
                unordered_map_type,
                children_are_key_value,
                "size=5 {",
                "hello",
                "world",
                "this",
                "is",
                "me",
            ],
        )

        unordered_mmap_type = (
            "std::unordered_multimap<int, std::basic_string<char, std::char_traits<char>, std::allocator<char>>, std::hash<int>, std::equal_to<int>, std::allocator<std::pair<int const, std::basic_string<char, std::char_traits<char>, std::allocator<char>>>>>"
            if self.getDebugInfo() == "pdb"
            else "UnorderedMultiMap"
        )
        self.look_for_content_and_continue(
            "mmap",
            [
                unordered_mmap_type,
                children_are_key_value,
                "size=6 {",
                "first = 3",
                'second = "this"',
                "first = 2",
                'second = "hello"',
            ],
        )

        ints_unordered_set = (
            "std::unordered_set<int, std::hash<int>, std::equal_to<int>, std::allocator<int>>"
            if self.getDebugInfo() == "pdb"
            else "IntsUnorderedSet"
        )
        self.look_for_content_and_continue(
            "iset",
            [
                ints_unordered_set,
                "size=5 {",
                r"\[\d\] = 5",
                r"\[\d\] = 3",
                r"\[\d\] = 2",
            ],
        )

        strings_unordered_set = (
            "std::unordered_set<std::basic_string<char, std::char_traits<char>, std::allocator<char>>, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>, std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>>"
            if self.getDebugInfo() == "pdb"
            else "StringsUnorderedSet"
        )
        self.look_for_content_and_continue(
            "sset",
            [
                strings_unordered_set,
                "size=5 {",
                r'\[\d\] = "is"',
                r'\[\d\] = "world"',
                r'\[\d\] = "hello"',
            ],
        )

        ints_unordered_mset = (
            "std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, std::allocator<int>>"
            if self.getDebugInfo() == "pdb"
            else "IntsUnorderedMultiSet"
        )
        self.look_for_content_and_continue(
            "imset",
            [
                ints_unordered_mset,
                "size=6 {",
                "(\\[\\d\\] = 3(\\n|.)+){3}",
                r"\[\d\] = 2",
                r"\[\d\] = 1",
            ],
        )

        strings_unordered_mset = (
            "std::unordered_multiset<std::basic_string<char, std::char_traits<char>, std::allocator<char>>, std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>, std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>, std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>>"
            if self.getDebugInfo() == "pdb"
            else "StringsUnorderedMultiSet"
        )
        self.look_for_content_and_continue(
            "smset",
            [
                strings_unordered_mset,
                "size=5 {",
                '(\\[\\d\\] = "is"(\\n|.)+){2}',
                '(\\[\\d\\] = "world"(\\n|.)+){2}',
            ],
        )

    def look_for_content_and_continue(self, var_name, patterns):
        self.expect(("frame variable %s" % var_name), ordered=False, patterns=patterns)
        self.expect(("frame variable %s" % var_name), ordered=False, patterns=patterns)
        self.runCmd("continue")

    @add_test_categories(["libstdcxx"])
    def test_with_run_command_libstdcpp(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_with_run_command()

    @add_test_categories(["libstdcxx"])
    def test_with_run_command_libstdcxx_debug(self):
        self.build(
            dictionary={"USE_LIBSTDCPP": 1, "CXXFLAGS_EXTRAS": "-D_GLIBCXX_DEBUG"}
        )
        self.do_test_with_run_command()

    @add_test_categories(["libc++"])
    def test_with_run_command_libcpp(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_with_run_command()

    @add_test_categories(["msvcstl"])
    def test_with_run_command_msvcstl(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test_with_run_command()
