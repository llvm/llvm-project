"""
Test some target commands: create, list, select, variable.
"""

import os
import stat
import tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class targetCommandTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for our breakpoints.
        self.line_b = line_number('b.c', '// Set break point at this line.')
        self.line_c = line_number('c.c', '// Set break point at this line.')

    def buildB(self):
        db = {'C_SOURCES': 'b.c', 'EXE': self.getBuildArtifact('b.out')}
        self.build(dictionary=db)
        self.addTearDownCleanup(dictionary=db)

    def buildAll(self):
        da = {'C_SOURCES': 'a.c', 'EXE': self.getBuildArtifact('a.out')}
        self.build(dictionary=da)
        self.addTearDownCleanup(dictionary=da)

        self.buildB()

        dc = {'C_SOURCES': 'c.c', 'EXE': self.getBuildArtifact('c.out')}
        self.build(dictionary=dc)
        self.addTearDownCleanup(dictionary=dc)

    def test_target_command(self):
        """Test some target commands: create, list, select."""
        self.buildAll()
        self.do_target_command()

    @expectedFailureDarwin(archs=["arm64", "arm64e"]) # <rdar://problem/37773624>
    def test_target_variable_command(self):
        """Test 'target variable' command before and after starting the inferior."""
        d = {'C_SOURCES': 'globals.c', 'EXE': self.getBuildArtifact('globals')}
        self.build(dictionary=d)
        self.addTearDownCleanup(dictionary=d)

        self.do_target_variable_command('globals')

    @expectedFailureDarwin(archs=["arm64", "arm64e"]) # <rdar://problem/37773624>
    def test_target_variable_command_no_fail(self):
        """Test 'target variable' command before and after starting the inferior."""
        d = {'C_SOURCES': 'globals.c', 'EXE': self.getBuildArtifact('globals')}
        self.build(dictionary=d)
        self.addTearDownCleanup(dictionary=d)

        self.do_target_variable_command_no_fail('globals')

    def do_target_command(self):
        """Exercise 'target create', 'target list', 'target select' commands."""
        exe_a = self.getBuildArtifact("a.out")
        exe_b = self.getBuildArtifact("b.out")
        exe_c = self.getBuildArtifact("c.out")

        self.runCmd("target list")
        output = self.res.GetOutput()
        if output.startswith("No targets"):
            # We start from index 0.
            base = 0
        else:
            # Find the largest index of the existing list.
            import re
            pattern = re.compile("target #(\d+):")
            for line in reversed(output.split(os.linesep)):
                match = pattern.search(line)
                if match:
                    # We will start from (index + 1) ....
                    base = int(match.group(1), 10) + 1
                    self.trace("base is:", base)
                    break

        self.runCmd("target create " + exe_a, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target create " + exe_b, CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, 'b.c', self.line_b, num_expected_locations=1, loc_exact=True)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target create " + exe_c, CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, 'c.c', self.line_c, num_expected_locations=1, loc_exact=True)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target list")

        self.runCmd("target select %d" % base)
        self.runCmd("thread backtrace")

        self.runCmd("target select %d" % (base + 2))
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stop reason = breakpoint' ,'c.c:%d' % self.line_c
                             ])

        self.runCmd("target select %d" % (base + 1))
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stop reason = breakpoint', 'b.c:%d' % self.line_b
                             ])

        self.runCmd("target list")

    @no_debug_info_test
    def test_target_create_invalid_arch(self):
        exe = self.getBuildArtifact("a.out")
        self.expect("target create {} --arch doesntexist".format(exe), error=True,
                    patterns=["error: invalid triple 'doesntexist'"])

    @no_debug_info_test
    def test_target_create_platform(self):
        self.buildB()
        exe = self.getBuildArtifact("b.out")
        self.expect("target create {} --platform host".format(exe))

    @no_debug_info_test
    def test_target_create_unsupported_platform(self):
        yaml = os.path.join(self.getSourceDir(), "bogus.yaml")
        exe = self.getBuildArtifact("bogus")
        self.yaml2obj(yaml, exe)
        self.expect("target create {}".format(exe), error=True,
                    patterns=['error: no matching platforms found for this file'])

    @no_debug_info_test
    def test_target_create_invalid_platform(self):
        self.buildB()
        exe = self.getBuildArtifact("b.out")
        self.expect("target create {} --platform doesntexist".format(exe), error=True,
                    patterns=['error: unable to find a plug-in for the platform named "doesntexist"'])

    def do_target_variable_command(self, exe_name):
        """Exercise 'target variable' command before and after starting the inferior."""
        self.runCmd("file " + self.getBuildArtifact(exe_name),
                    CURRENT_EXECUTABLE_SET)

        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])
        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])

        self.runCmd("b main")
        self.runCmd("run")

        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])
        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])

        self.runCmd("c")

        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])
        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])

    def do_target_variable_command_no_fail(self, exe_name):
        """Exercise 'target variable' command before and after starting the inferior."""
        self.runCmd("file " + self.getBuildArtifact(exe_name),
                    CURRENT_EXECUTABLE_SET)

        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])
        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])

        self.runCmd("b main")
        self.runCmd("run")

        # New feature: you don't need to specify the variable(s) to 'target vaiable'.
        # It will find all the global and static variables in the current
        # compile unit.
        self.expect("target variable",
                    ordered=False,
                    substrs=['my_global_char',
                             'my_static_int',
                             'my_global_str',
                             'my_global_str_ptr',
                             ])

        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])
        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])

    @no_debug_info_test
    def test_target_stop_hook_disable_enable(self):
        self.buildB()
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)

        self.expect("target stop-hook disable 1", error=True, substrs=['unknown stop hook id: "1"'])
        self.expect("target stop-hook disable blub", error=True, substrs=['invalid stop hook id: "blub"'])
        self.expect("target stop-hook enable 1", error=True, substrs=['unknown stop hook id: "1"'])
        self.expect("target stop-hook enable blub", error=True, substrs=['invalid stop hook id: "blub"'])

    @no_debug_info_test
    def test_target_stop_hook_delete(self):
        self.buildB()
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)

        self.expect("target stop-hook delete 1", error=True, substrs=['unknown stop hook id: "1"'])
        self.expect("target stop-hook delete blub", error=True, substrs=['invalid stop hook id: "blub"'])

    @no_debug_info_test
    def test_target_list_args(self):
        self.expect("target list blub", error=True,
                    substrs=["the 'target list' command takes no arguments"])

    @no_debug_info_test
    def test_target_select_no_index(self):
        self.expect("target select", error=True,
                    substrs=["'target select' takes a single argument: a target index"])

    @no_debug_info_test
    def test_target_select_invalid_index(self):
        self.runCmd("target delete --all")
        self.expect("target select 0", error=True,
                    substrs=["index 0 is out of range since there are no active targets"])
        self.buildB()
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)
        self.expect("target select 1", error=True,
                    substrs=["index 1 is out of range, valid target indexes are 0 - 0"])


    @no_debug_info_test
    def test_target_create_multiple_args(self):
        self.expect("target create a b", error=True,
                    substrs=["'target create' takes exactly one executable path"])

    @skipIfWindowsAndNonEnglish
    @no_debug_info_test
    def test_target_create_nonexistent_core_file(self):
        self.expect("target create -c doesntexist", error=True,
                    patterns=["Cannot open 'doesntexist'", ": (No such file or directory|The system cannot find the file specified)"])

    # Write only files don't seem to be supported on Windows.
    @skipIfWindows
    @no_debug_info_test
    def test_target_create_unreadable_core_file(self):
        tf = tempfile.NamedTemporaryFile()
        os.chmod(tf.name, stat.S_IWRITE)
        self.expect("target create -c '" + tf.name + "'", error=True,
                    substrs=["Cannot open '", "': Permission denied"])

    @skipIfWindowsAndNonEnglish
    @no_debug_info_test
    def test_target_create_nonexistent_sym_file(self):
        self.expect("target create -s doesntexist doesntexisteither", error=True,
                    patterns=["Cannot open '", ": (No such file or directory|The system cannot find the file specified)"])

    @skipIfWindows
    @no_debug_info_test
    def test_target_create_invalid_core_file(self):
        invalid_core_path = os.path.join(self.getSourceDir(), "invalid_core_file")
        self.expect("target create -c '" + invalid_core_path + "'", error=True,
                    substrs=["Unable to find process plug-in for core file '"])


    # Write only files don't seem to be supported on Windows.
    @skipIfWindows
    @no_debug_info_test
    def test_target_create_unreadable_sym_file(self):
        tf = tempfile.NamedTemporaryFile()
        os.chmod(tf.name, stat.S_IWRITE)
        self.expect("target create -s '" + tf.name + "' no_exe", error=True,
                    substrs=["Cannot open '", "': Permission denied"])

    @no_debug_info_test
    def test_target_delete_all(self):
        self.buildAll()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)
        self.expect("target delete --all")
        self.expect("target list", substrs=["No targets."])

    @no_debug_info_test
    def test_target_delete_by_index(self):
        self.buildAll()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("file " + self.getBuildArtifact("c.out"), CURRENT_EXECUTABLE_SET)
        self.expect("target delete 3", error=True,
                    substrs=["target index 3 is out of range, valid target indexes are 0 - 2"])

        self.runCmd("target delete 1")
        self.expect("target list", matching=False, substrs=["b.out"])
        self.runCmd("target delete 1")
        self.expect("target list", matching=False, substrs=["c.out"])

        self.expect("target delete 1", error=True,
                    substrs=["target index 1 is out of range, the only valid index is 0"])

        self.runCmd("target delete 0")
        self.expect("target list", matching=False, substrs=["a.out"])

        self.expect("target delete 0", error=True, substrs=["no targets to delete"])
        self.expect("target delete 1", error=True, substrs=["no targets to delete"])

    @no_debug_info_test
    def test_target_delete_by_index_multiple(self):
        self.buildAll()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("file " + self.getBuildArtifact("c.out"), CURRENT_EXECUTABLE_SET)

        self.expect("target delete 0 1 2 3", error=True,
                    substrs=["target index 3 is out of range, valid target indexes are 0 - 2"])
        self.expect("target list", substrs=["a.out", "b.out", "c.out"])

        self.runCmd("target delete 0 1 2")
        self.expect("target list", matching=False, substrs=["a.out", "c.out"])

    @no_debug_info_test
    def test_target_delete_selected(self):
        self.buildAll()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("file " + self.getBuildArtifact("c.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("target select 1")
        self.runCmd("target delete")
        self.expect("target list", matching=False, substrs=["b.out"])
        self.runCmd("target delete")
        self.runCmd("target delete")
        self.expect("target list", substrs=["No targets."])
        self.expect("target delete", error=True, substrs=["no target is currently selected"])

    @no_debug_info_test
    def test_target_modules_search_paths_clear(self):
        self.buildB()
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("target modules search-paths add foo bar")
        self.runCmd("target modules search-paths add foz baz")
        self.runCmd("target modules search-paths clear")
        self.expect("target list", matching=False, substrs=["bar", "baz"])

    @no_debug_info_test
    def test_target_modules_search_paths_query(self):
        self.buildB()
        self.runCmd("file " + self.getBuildArtifact("b.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("target modules search-paths add foo bar")
        self.expect("target modules search-paths query foo", substrs=["bar"])
        # Query something that doesn't exist.
        self.expect("target modules search-paths query faz", substrs=["faz"])

        # Invalid arguments.
        self.expect("target modules search-paths query faz baz", error=True,
                    substrs=["query requires one argument"])

    @no_debug_info_test
    def test_target_modules_type(self):
        self.buildB()
        self.runCmd("file " + self.getBuildArtifact("b.out"),
                    CURRENT_EXECUTABLE_SET)
        self.expect("target modules lookup --type int",
                    substrs=["1 match found", 'name = "int"'])
