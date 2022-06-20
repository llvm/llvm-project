"""Test the LLDB module cache funcionality."""

import glob
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import time


class ModuleCacheTestcaseBSD(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number in a(int) to break at.
        self.line_a = line_number(
            'a.c', '// Set file and line breakpoint inside a().')
        self.line_b = line_number(
            'b.c', '// Set file and line breakpoint inside b().')
        self.line_c = line_number(
            'c.c', '// Set file and line breakpoint inside c().')
        self.cache_dir = os.path.join(self.getBuildDir(), 'lldb-module-cache')
        # Set the lldb module cache directory to a directory inside the build
        # artifacts directory so no other tests are interfered with.
        self.runCmd('settings set symbols.lldb-index-cache-path "%s"' % (self.cache_dir))
        self.runCmd('settings set symbols.enable-lldb-index-cache true')
        self.build()


    def get_module_cache_files(self, basename):
        module_cache_glob = os.path.join(self.cache_dir, "llvmcache-*%s*symtab*" % (basename))
        return glob.glob(module_cache_glob)


    # Requires no dSYM, so we let the Makefile make the right stuff for us
    @no_debug_info_test
    @skipUnlessDarwin
    def test(self):
        """
            This test has been modified to make sure .o files that don't have
            UUIDs are not cached after discovering build systems that play with
            modification times of .o files that the modification times are not
            unique enough to ensure the .o file within the .a file are the right
            files as this was causing older cache files to be accepted for new
            updated .o files.

            ELF .o files do calculate a UUID from the contents of the file,
            which is expensive, but no one loads .o files into a debug sessions
            when using ELF files. Mach-o .o files do not have UUID values and do
            no calculate one as they _are_ used during debug sessions when no
            dSYM file is generated. If we can find a way to uniquely and cheaply
            create UUID values for mach-o .o files in the future, this test will
            be updated to test this functionality. This test will now make sure
            there are no cache entries for any .o files in BSD archives.

            The old test case description is below in case we enable caching for
            .o files again:

            Test module cache functionality for bsd archive object files.

            This will test that if we enable the module cache, we have a
            corresponding cache entry for the .o files in libfoo.a.

            The static library has two entries for "a.o":
            - one from a.c
            - one from c.c which had c.o renamed to a.o and then put into the
              libfoo.a as an extra .o file with different contents from the
              original a.o

            We do this to test that we can correctly cache duplicate .o files
            that appear in .a files.

            This test only works on darwin because of the way DWARF is stored
            where the debug map will refer to .o files inside of .a files.
        """
        exe = self.getBuildArtifact("a.out")

        # Create a module with no depedencies.
        target = self.createTestTarget(load_dependent_modules=False)

        self.runCmd('breakpoint set -f a.c -l %d' % (self.line_a))
        self.runCmd('breakpoint set -f b.c -l %d' % (self.line_b))
        self.runCmd('breakpoint set -f c.c -l %d' % (self.line_c))

        # Get the executable module and get the number of symbols to make
        # sure the symbol table gets parsed and cached. The module cache is
        # enabled in the setUp() function.
        main_module = target.GetModuleAtIndex(0)
        self.assertTrue(main_module.IsValid())
        # Make sure the symbol table gets loaded and cached
        main_module.GetNumSymbols()
        a_o_cache_files = self.get_module_cache_files("libfoo.a(a.o)")
        b_o_cache_files = self.get_module_cache_files("libfoo.a(b.o)")


        # We expect the directory for a.o to have two cache directories:
        # - 1 for the a.o with a earlier mod time
        # - 1 for the a.o that was renamed from c.o that should be 2 seconds older
        # self.assertEqual(len(a_o_cache_files), 2,
        #         "make sure there are two files in the module cache directory (%s) for libfoo.a(a.o)" % (self.cache_dir))
        # self.assertEqual(len(b_o_cache_files), 1,
        #         "make sure there are two files in the module cache directory (%s) for libfoo.a(b.o)" % (self.cache_dir))

        # We are no longer caching .o files in the lldb index cache. If we ever
        # re-enable this functionality, we can uncomment out the above lines of
        # code.
        self.assertEqual(len(a_o_cache_files), 0,
                "make sure there are no files in the module cache directory (%s) for libfoo.a(a.o)" % (self.cache_dir))
        self.assertEqual(len(b_o_cache_files), 0,
                "make sure there are no files in the module cache directory (%s) for libfoo.a(b.o)" % (self.cache_dir))
