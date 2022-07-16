"""Test the LLDB module cache funcionality for universal mach-o files."""

import glob
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import time


class ModuleCacheTestcaseUniversal(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number in a(int) to break at.
        self.cache_dir = os.path.join(self.getBuildDir(), 'lldb-module-cache')
        # Set the lldb module cache directory to a directory inside the build
        # artifacts directory so no other tests are interfered with.
        self.runCmd('settings set symbols.lldb-index-cache-path "%s"' % (self.cache_dir))
        self.runCmd('settings set symbols.enable-lldb-index-cache true')

    def get_module_cache_files(self, basename):
        module_file_glob = os.path.join(self.cache_dir, "llvmcache-*%s*" % (basename))
        return glob.glob(module_file_glob)


    # Doesn't depend on any specific debug information.
    @no_debug_info_test
    def test(self):
        """
            Test module cache functionality for a universal mach-o files.

            This will test that if we enable the module cache, we can create
            lldb module caches for each slice of a universal mach-o file and
            they will each have a unique directory.
        """
        exe_basename = "testit"
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "universal.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        exe = self.getBuildArtifact(exe_basename)
        self.yaml2obj(yaml_path, exe)
        self.assertTrue(os.path.exists(exe))
        # Create a module with no depedencies.
        self.runCmd('target create -d --arch x86_64 %s' % (exe))
        self.runCmd('image dump symtab %s' % (exe_basename))
        self.runCmd('target create -d --arch arm64 %s' % (exe))
        self.runCmd('image dump symtab %s' % (exe_basename))

        cache_files = self.get_module_cache_files(exe_basename)

        self.assertEqual(len(cache_files), 2,
                "make sure there are two files in the module cache directory (%s) for %s" % (self.cache_dir, exe_basename))
