"""
They may be cases where an expression will import SourceLocation and if the
SourceLocation ends up with a FileID that is a built-in we need to copy that
buffer over correctly.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestImportBuiltinFileID(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @add_test_categories(["gmodules"])
    def test_import_builtin_fileid(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.m", False))

        self.expect("expr int (*DBG_CGImageGetRenderingIntent)(void *) = ((int (*)(void *))CGImageGetRenderingIntent); DBG_CGImageGetRenderingIntent((void *)0x00000000000000);", 
                substrs=['$0 = 0'])
