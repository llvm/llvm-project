"""
Test lldb core component: SourceManager.

Test cases:

o test_display_source_python:
  Test display of source using the SBSourceManager API.
o test_modify_source_file_while_debugging:
  Test the caching mechanism of the source manager.
"""

import os
import stat

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def ansi_underline_surround_regex(inner_regex_text):
    # return re.compile(r"\[4m%s\[0m" % inner_regex_text)
    return "4.+\033\\[4m%s\033\\[0m" % inner_regex_text

def ansi_color_surround_regex(inner_regex_text):
    return "\033\\[3[0-7]m%s\033\\[0m" % inner_regex_text

class SourceManagerTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.file = self.getBuildArtifact("main-copy.c")
        self.line = line_number("main.c", '// Set break point at this line.')

    def get_expected_stop_column_number(self):
        """Return the 1-based column number of the first non-whitespace
        character in the breakpoint source line."""
        stop_line = get_line(self.file, self.line)
        # The number of spaces that must be skipped to get to the first non-
        # whitespace character --- where we expect the debugger breakpoint
        # column to be --- is equal to the number of characters that get
        # stripped off the front when we lstrip it, plus one to specify
        # the character column after the initial whitespace.
        return len(stop_line) - len(stop_line.lstrip()) + 1

    def do_display_source_python_api(self, use_color, needle_regex, highlight_source=False):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        args = None
        envp = None
        process = target.LaunchSimple(
            args, envp, self.get_process_working_directory())
        self.assertIsNotNone(process)

        #
        # Exercise Python APIs to display source lines.
        #

        # Setup whether we should use ansi escape sequences, including color
        # and styles such as underline.
        self.dbg.SetUseColor(use_color)
        # Disable syntax highlighting if needed.

        self.runCmd("settings set highlight-source " + str(highlight_source).lower())

        filespec = lldb.SBFileSpec(self.file, False)
        source_mgr = self.dbg.GetSourceManager()
        # Use a string stream as the destination.
        stream = lldb.SBStream()
        column = self.get_expected_stop_column_number()
        context_before = 2
        context_after = 2
        current_line_prefix = "=>"
        source_mgr.DisplaySourceLinesWithLineNumbersAndColumn(
            filespec, self.line, column, context_before, context_after,
            current_line_prefix, stream)

        #    2
        #    3    int main(int argc, char const *argv[]) {
        # => 4        printf("Hello world.\n"); // Set break point at this line.
        #    5        return 0;
        #    6    }
        self.expect(stream.GetData(), "Source code displayed correctly:\n" + stream.GetData(),
                    exe=False,
                    patterns=['=>', '%d.*Hello world' % self.line,
                              needle_regex])

        # Boundary condition testings for SBStream().  LLDB should not crash!
        stream.Print(None)
        stream.RedirectToFile(None, True)

    @add_test_categories(['pyapi'])
    def test_display_source_python_dumb_terminal(self):
        """Test display of source using the SBSourceManager API, using a
        dumb terminal and thus no color support (the default)."""
        use_color = False
        self.do_display_source_python_api(use_color, r"\s+\^")

    @add_test_categories(['pyapi'])
    def test_display_source_python_ansi_terminal(self):
        """Test display of source using the SBSourceManager API, using a
        dumb terminal and thus no color support (the default)."""
        use_color = True
        underline_regex = ansi_underline_surround_regex(r"printf")
        self.do_display_source_python_api(use_color, underline_regex)

    @add_test_categories(['pyapi'])
    def test_display_source_python_ansi_terminal_syntax_highlighting(self):
        """Test display of source using the SBSourceManager API and check for
        the syntax highlighted output"""
        use_color = True
        syntax_highlighting = True;

        # Just pick 'int' as something that should be colored.
        color_regex = ansi_color_surround_regex("int")
        self.do_display_source_python_api(use_color, color_regex, syntax_highlighting)

        # Same for 'char'.
        color_regex = ansi_color_surround_regex("char")
        self.do_display_source_python_api(use_color, color_regex, syntax_highlighting)

        # Test that we didn't color unrelated identifiers.
        self.do_display_source_python_api(use_color, r" main\(", syntax_highlighting)
        self.do_display_source_python_api(use_color, r"\);", syntax_highlighting)

    def test_move_and_then_display_source(self):
        """Test that target.source-map settings work by moving main.c to hidden/main.c."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Move main.c to hidden/main.c.
        hidden = self.getBuildArtifact("hidden")
        lldbutil.mkdir_p(hidden)
        main_c_hidden = os.path.join(hidden, "main-copy.c")
        os.rename(self.file, main_c_hidden)

        # Set source remapping with invalid replace path and verify we get an
        # error
        self.expect(
            "settings set target.source-map /a/b/c/d/e /q/r/s/t/u",
            error=True,
            substrs=['''error: the replacement path doesn't exist: "/q/r/s/t/u"'''])

        # 'make -C' has resolved current directory to its realpath form.
        builddir_real = os.path.realpath(self.getBuildDir())
        hidden_real = os.path.realpath(hidden)
        # Set target.source-map settings.
        self.runCmd("settings set target.source-map %s %s" %
                    (builddir_real, hidden_real))
        # And verify that the settings work.
        self.expect("settings show target.source-map",
                    substrs=[builddir_real, hidden_real])

        # Display main() and verify that the source mapping has been kicked in.
        self.expect("source list -n main", SOURCE_DISPLAYED_CORRECTLY,
                    substrs=['Hello world'])

    @skipIf(oslist=["windows"], bugnumber="llvm.org/pr44431")
    def test_modify_source_file_while_debugging(self):
        """Modify a source file while debugging the executable."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main-copy.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'main-copy.c:%d' % self.line,
                             'stop reason = breakpoint'])

        # Display some source code.
        self.expect(
            "source list -f main-copy.c -l %d" %
            self.line,
            SOURCE_DISPLAYED_CORRECTLY,
            substrs=['Hello world'])

        # Do the same thing with a file & line spec:
        self.expect(
            "source list -y main-copy.c:%d" %
            self.line,
            SOURCE_DISPLAYED_CORRECTLY,
            substrs=['Hello world'])

        # The '-b' option shows the line table locations from the debug information
        # that indicates valid places to set source level breakpoints.

        # The file to display is implicit in this case.
        self.runCmd("source list -l %d -c 3 -b" % self.line)
        output = self.res.GetOutput().splitlines()[0]

        # If the breakpoint set command succeeded, we should expect a positive number
        # of breakpoints for the current line, i.e., self.line.
        import re
        m = re.search('^\[(\d+)\].*// Set break point at this line.', output)
        if not m:
            self.fail("Fail to display source level breakpoints")
        self.assertTrue(int(m.group(1)) > 0)

        # Read the main.c file content.
        with io.open(self.file, 'r', newline='\n') as f:
            original_content = f.read()
            if self.TraceOn():
                print("original content:", original_content)

        # Modify the in-memory copy of the original source code.
        new_content = original_content.replace('Hello world', 'Hello lldb', 1)

        # Modify the source code file.
        # If the source was read only, the copy will also be read only.
        # Run "chmod u+w" on it first so we can modify it.
        statinfo = os.stat(self.file)
        os.chmod(self.file, statinfo.st_mode | stat.S_IWUSR)

        with io.open(self.file, 'w', newline='\n') as f:
            time.sleep(1)
            f.write(new_content)
            if self.TraceOn():
                print("new content:", new_content)
                print(
                    "os.path.getmtime() after writing new content:",
                    os.path.getmtime(self.file))

        # Display the source code again.  We should see the updated line.
        self.expect(
            "source list -f main-copy.c -l %d" %
            self.line,
            SOURCE_DISPLAYED_CORRECTLY,
            substrs=['Hello lldb'])

    def test_set_breakpoint_with_absolute_path(self):
        self.build()
        hidden = self.getBuildArtifact("hidden")
        lldbutil.mkdir_p(hidden)
        # 'make -C' has resolved current directory to its realpath form.
        builddir_real = os.path.realpath(self.getBuildDir())
        hidden_real = os.path.realpath(hidden)
        self.runCmd("settings set target.source-map %s %s" %
                    (builddir_real, hidden_real))

        exe = self.getBuildArtifact("a.out")
        main = os.path.join(builddir_real, "hidden", "main-copy.c")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, main, self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'main-copy.c:%d' % self.line,
                             'stop reason = breakpoint'])

    def test_artificial_source_location(self):
        src_file = 'artificial_location.c'
        d = {'C_SOURCES': src_file }
        self.build(dictionary=d)

        lldbutil.run_to_source_breakpoint(
            self, 'main',
            lldb.SBFileSpec(src_file, False))

        self.expect("run", RUN_SUCCEEDED,
                    substrs=['stop reason = breakpoint', '%s:%d' % (src_file,0),
                             'Note: this address is compiler-generated code in '
                             'function', 'that has no source code associated '
                             'with it.'])

