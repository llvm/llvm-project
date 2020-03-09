"""
Test breakpoint command with AT_comp_dir set to symbolic link.
"""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


_EXE_NAME = 'CompDirSymLink'  # Must match Makefile
_SRC_FILE = 'relative.cpp'
_COMP_DIR_SYM_LINK_PROP = 'plugin.symbol-file.dwarf.comp-dir-symlink-paths'


class CompDirSymLinkTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number(
            os.path.join(self.getSourceDir(), "main.cpp"),
            '// Set break point at this line.')

    @skipIf(hostoslist=["windows"])
    def test_symlink_paths_set(self):
        pwd_symlink = self.create_src_symlink()
        self.doBuild(pwd_symlink)
        self.runCmd(
            "settings set %s %s" %
            (_COMP_DIR_SYM_LINK_PROP, pwd_symlink))
        src_path = self.getBuildArtifact(_SRC_FILE)
        lldbutil.run_break_set_by_file_and_line(self, src_path, self.line)

    @skipIf(hostoslist=no_match(["linux"]))
    def test_symlink_paths_set_procselfcwd(self):
        os.chdir(self.getBuildDir())
        pwd_symlink = '/proc/self/cwd'
        self.doBuild(pwd_symlink)
        self.runCmd(
            "settings set %s %s" %
            (_COMP_DIR_SYM_LINK_PROP, pwd_symlink))
        src_path = self.getBuildArtifact(_SRC_FILE)
        # /proc/self/cwd points to a realpath form of current directory.
        src_path = os.path.realpath(src_path)
        lldbutil.run_break_set_by_file_and_line(self, src_path, self.line)

    @skipIf(hostoslist=["windows"])
    def test_symlink_paths_unset(self):
        pwd_symlink = self.create_src_symlink()
        self.doBuild(pwd_symlink)
        self.runCmd('settings clear ' + _COMP_DIR_SYM_LINK_PROP)
        src_path = self.getBuildArtifact(_SRC_FILE)
        self.assertRaises(
            AssertionError,
            lldbutil.run_break_set_by_file_and_line,
            self,
            src_path,
            self.line)

    def create_src_symlink(self):
        pwd_symlink = self.getBuildArtifact('pwd_symlink')
        if os.path.exists(pwd_symlink):
            os.unlink(pwd_symlink)
        os.symlink(self.getBuildDir(), pwd_symlink)
        self.addTearDownHook(lambda: os.remove(pwd_symlink))
        return pwd_symlink

    def doBuild(self, pwd_symlink):
        self.build(None, None, {'PWD': pwd_symlink})

        exe = self.getBuildArtifact(_EXE_NAME)
        self.runCmd('file ' + exe, CURRENT_EXECUTABLE_SET)
