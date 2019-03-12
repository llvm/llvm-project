# -*- coding: utf-8 -*-
import subprocess
import tempfile
import shutil
import errno
import os

import pexpect
from . import PexpectTestCase

import pytest


class TestCaseWhich(PexpectTestCase.PexpectTestCase):
    " Tests for pexpect.which(). "

    def test_which_finds_ls(self):
        " which() can find ls(1). "
        exercise = pexpect.which("ls")
        assert exercise is not None
        assert exercise.startswith('/')

    def test_path_from_env(self):
        " executable found from optional env argument "
        bin_name = 'pexpect-test-path-from-env'
        tempdir = tempfile.mkdtemp()
        try:
            bin_path = os.path.join(tempdir, bin_name)
            with open(bin_path, 'w') as f:
                f.write('# test file not to be run')
            try:
                os.chmod(bin_path, 0o700)
                found_path = pexpect.which(bin_name, env={'PATH': tempdir})
            finally:
                os.remove(bin_path)
            self.assertEqual(bin_path, found_path)
        finally:
            os.rmdir(tempdir)

    def test_os_defpath_which(self):
        " which() finds an executable in $PATH and returns its abspath. "

        bin_dir = tempfile.mkdtemp()
        temp_obj = tempfile.NamedTemporaryFile(
            suffix=u'.sh', prefix=u'ǝpoɔıun-',
            dir=bin_dir, delete=False)
        bin_path = temp_obj.name
        fname = os.path.basename(temp_obj.name)
        save_path = os.environ['PATH']
        save_defpath = os.defpath

        try:
            # setup
            os.environ['PATH'] = ''
            os.defpath = bin_dir
            with open(bin_path, 'w') as fp:
                pass

            # given non-executable,
            os.chmod(bin_path, 0o400)

            # exercise absolute and relative,
            assert pexpect.which(bin_path) is None
            assert pexpect.which(fname) is None

            # given executable,
            os.chmod(bin_path, 0o700)

            # exercise absolute and relative,
            assert pexpect.which(bin_path) == bin_path
            assert pexpect.which(fname) == bin_path

        finally:
            # restore,
            os.environ['PATH'] = save_path
            os.defpath = save_defpath

            # destroy scratch files and folders,
            if os.path.exists(bin_path):
                os.unlink(bin_path)
            if os.path.exists(bin_dir):
                os.rmdir(bin_dir)

    def test_path_search_which(self):
        " which() finds an executable in $PATH and returns its abspath. "
        fname = 'gcc'
        bin_dir = tempfile.mkdtemp()
        bin_path = os.path.join(bin_dir, fname)
        save_path = os.environ['PATH']
        try:
            # setup
            os.environ['PATH'] = bin_dir
            with open(bin_path, 'w') as fp:
                pass

            # given non-executable,
            os.chmod(bin_path, 0o400)

            # exercise absolute and relative,
            assert pexpect.which(bin_path) is None
            assert pexpect.which(fname) is None

            # given executable,
            os.chmod(bin_path, 0o700)

            # exercise absolute and relative,
            assert pexpect.which(bin_path) == bin_path
            assert pexpect.which(fname) == bin_path

        finally:
            # restore,
            os.environ['PATH'] = save_path

            # destroy scratch files and folders,
            if os.path.exists(bin_path):
                os.unlink(bin_path)
            if os.path.exists(bin_dir):
                os.rmdir(bin_dir)

    def test_which_follows_symlink(self):
        " which() follows symlinks and returns its path. "
        fname = 'original'
        symname = 'extra-crispy'
        bin_dir = tempfile.mkdtemp()
        bin_path = os.path.join(bin_dir, fname)
        sym_path = os.path.join(bin_dir, symname)
        save_path = os.environ['PATH']
        try:
            # setup
            os.environ['PATH'] = bin_dir
            with open(bin_path, 'w') as fp:
                pass
            os.chmod(bin_path, 0o400)
            os.symlink(bin_path, sym_path)

            # should not be found because symlink points to non-executable
            assert pexpect.which(symname) is None

            # but now it should -- because it is executable
            os.chmod(bin_path, 0o700)
            assert pexpect.which(symname) == sym_path

        finally:
            # restore,
            os.environ['PATH'] = save_path

            # destroy scratch files, symlinks, and folders,
            if os.path.exists(sym_path):
                os.unlink(sym_path)
            if os.path.exists(bin_path):
                os.unlink(bin_path)
            if os.path.exists(bin_dir):
                os.rmdir(bin_dir)

    def test_which_should_not_match_folders(self):
        " Which does not match folders, even though they are executable. "
        # make up a path and insert a folder that is 'executable', a naive
        # implementation might match (previously pexpect versions 3.2 and
        # sh versions 1.0.8, reported by @lcm337.)
        fname = 'g++'
        bin_dir = tempfile.mkdtemp()
        bin_dir2 = os.path.join(bin_dir, fname)
        save_path = os.environ['PATH']
        try:
            os.environ['PATH'] = bin_dir
            os.mkdir(bin_dir2, 0o755)
            # should not be found because it is not executable *file*,
            # but rather, has the executable bit set, as a good folder
            # should -- it should not be returned because it fails isdir()
            exercise = pexpect.which(fname)
            assert exercise is None

        finally:
            # restore,
            os.environ['PATH'] = save_path
            # destroy scratch folders,
            for _dir in (bin_dir2, bin_dir,):
                if os.path.exists(_dir):
                    os.rmdir(_dir)

    def test_which_should_match_other_group_user(self):
        " which() returns executables by other, group, and user ownership. "
        # create an executable and test that it is found using which() for
        # each of the 'other', 'group', and 'user' permission bits.
        fname = 'g77'
        bin_dir = tempfile.mkdtemp()
        bin_path = os.path.join(bin_dir, fname)
        save_path = os.environ['PATH']
        try:
            # setup
            os.environ['PATH'] = bin_dir

            # an interpreted script requires the ability to read,
            # whereas a binary program requires only to be executable.
            #
            # to gain access to a binary program, we make a copy of
            # the existing system program echo(1).
            bin_echo = None
            for pth in ('/bin/echo', '/usr/bin/echo'):
                if os.path.exists(pth):
                    bin_echo = pth
                    break
            bin_which = None
            for pth in ('/bin/which', '/usr/bin/which'):
                if os.path.exists(pth):
                    bin_which = pth
                    break
            if not bin_echo or not bin_which:
                pytest.skip('needs `echo` and `which` binaries')
            shutil.copy(bin_echo, bin_path)
            isroot = os.getuid() == 0
            for should_match, mode in (
                # note that although the file may have matching 'group' or
                # 'other' executable permissions, it is *not* executable
                # because the current uid is the owner of the file -- which
                # takes precedence
                (False,  0o000),   # ----------, no
                (isroot, 0o001),   # ---------x, no
                (isroot, 0o010),   # ------x---, no
                (True,   0o100),   # ---x------, yes
                (False,  0o002),   # --------w-, no
                (False,  0o020),   # -----w----, no
                (False,  0o200),   # --w-------, no
                (isroot, 0o003),   # --------wx, no
                (isroot, 0o030),   # -----wx---, no
                (True,   0o300),   # --wx------, yes
                (False,  0o004),   # -------r--, no
                (False,  0o040),   # ----r-----, no
                (False,  0o400),   # -r--------, no
                (isroot, 0o005),   # -------r-x, no
                (isroot, 0o050),   # ----r-x---, no
                (True,   0o500),   # -r-x------, yes
                (False,  0o006),   # -------rw-, no
                (False,  0o060),   # ----rw----, no
                (False,  0o600),   # -rw-------, no
                (isroot, 0o007),   # -------rwx, no
                (isroot, 0o070),   # ----rwx---, no
                (True,   0o700),   # -rwx------, yes
                (isroot, 0o4001),  # ---S-----x, no
                (isroot, 0o4010),  # ---S--x---, no
                (True,   0o4100),  # ---s------, yes
                (isroot, 0o4003),  # ---S----wx, no
                (isroot, 0o4030),  # ---S-wx---, no
                (True,   0o4300),  # --ws------, yes
                (isroot, 0o2001),  # ------S--x, no
                (isroot, 0o2010),  # ------s---, no
                (True,   0o2100),  # ---x--S---, yes

            ):
                mode_str = '{0:0>4o}'.format(mode)

                # given file mode,
                os.chmod(bin_path, mode)

                # exercise whether we may execute
                can_execute = True
                try:
                    subprocess.Popen(fname).wait() == 0
                except OSError as err:
                    if err.errno != errno.EACCES:
                        raise
                    # permission denied
                    can_execute = False

                assert should_match == can_execute, (
                    should_match, can_execute, mode_str)

                # exercise whether which(1) would match
                proc = subprocess.Popen((bin_which, fname),
                                        env={'PATH': bin_dir},
                                        stdout=subprocess.PIPE)
                bin_which_match = bool(not proc.wait())
                assert should_match == bin_which_match, (
                    should_match, bin_which_match, mode_str)

                # finally, exercise pexpect's which(1) matches
                # the same.
                pexpect_match = bool(pexpect.which(fname))

                assert should_match == pexpect_match == bin_which_match, (
                    should_match, pexpect_match, bin_which_match, mode_str)

        finally:
            # restore,
            os.environ['PATH'] = save_path

            # destroy scratch files and folders,
            if os.path.exists(bin_path):
                os.unlink(bin_path)
            if os.path.exists(bin_dir):
                os.rmdir(bin_dir)
