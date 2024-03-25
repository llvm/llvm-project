"""
Test that re-running a process from within the same target
after rebuilding the a dynamic library flushes the scratch
TypeSystems tied to that process.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *


def isUbuntu18_04():
    """
    Check if the host OS is Ubuntu 18.04.
    Derived from `platform.freedesktop_os_release` in Python 3.10.
    """
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        if os.path.exists(path):
            with open(path) as f:
                contents = f.read()
            if "Ubuntu 18.04" in contents:
                return True

    return False


class TestRerunExprDylib(TestBase):
    @skipTestIfFn(isUbuntu18_04, bugnumber="rdar://103831050")
    @skipIfWindows
    def test(self):
        """
        Tests whether re-launching a process without destroying
        the owning target keeps invalid ASTContexts in the
        scratch AST's importer.

        We test this by:
        1. Evaluating an expression to import 'struct Foo' into
           the scratch AST
        2. Change the definition of 'struct Foo' and rebuild the dylib
        3. Re-launch the process
        4. Evaluate the same expression in (1). We expect to have only
           the latest definition of 'struct Foo' in the scratch AST.
        """

        DYLIB_NAME = "foo"
        FULL_DYLIB_NAME = "libfoo.dylib" if self.platformIsDarwin() else "libfoo.so"

        # Build libfoo.dylib
        self.build(
            dictionary={
                "DYLIB_CXX_SOURCES": "lib.cpp",
                "DYLIB_ONLY": "YES",
                "DYLIB_NAME": DYLIB_NAME,
                "USE_LIBDL": "1",
                "LD_EXTRAS": "-L.",
            }
        )

        # Build a.out
        self.build(
            dictionary={
                "EXE": "a.out",
                "CXX_SOURCES": "main.cpp",
                "USE_LIBDL": "1",
                "CXXFLAGS_EXTRAS": f'-DLIB_NAME=\\"{FULL_DYLIB_NAME}\\"',
                "LD_EXTRAS": "-L.",
            }
        )

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        target.BreakpointCreateBySourceRegex("dlclose", lldb.SBFileSpec("main.cpp"))
        target.BreakpointCreateBySourceRegex("return", lldb.SBFileSpec("main.cpp"))
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.expect_expr(
            "*foo",
            result_type="Foo",
            result_children=[ValueCheck(name="m_val", value="42")],
        )

        # Delete the dylib to force make to rebuild it.
        remove_file(self.getBuildArtifact(FULL_DYLIB_NAME))

        # Re-build libfoo.dylib
        self.build(
            dictionary={
                "DYLIB_CXX_SOURCES": "rebuild.cpp",
                "DYLIB_ONLY": "YES",
                "DYLIB_NAME": DYLIB_NAME,
                "USE_LIBDL": "1",
                "LD_EXTRAS": "-L.",
            }
        )

        # Rerun program within the same target
        process.Continue()
        process.Destroy()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.expect_expr(
            "*foo",
            result_type="Foo",
            result_children=[
                ValueCheck(
                    name="Base", children=[ValueCheck(name="m_base_val", value="42")]
                ),
                ValueCheck(name="m_derived_val", value="137"),
            ],
        )

        self.filecheck("target module dump ast", __file__)

        # The new definition 'struct Foo' is in the scratch AST
        # CHECK:      |-CXXRecordDecl {{.*}} struct Foo definition
        # CHECK:      | |-public 'Base'
        # CHECK-NEXT: | `-FieldDecl {{.*}} m_derived_val 'int'
        # CHECK-NEXT: `-CXXRecordDecl {{.*}} struct Base definition
        # CHECK:        `-FieldDecl {{.*}} m_base_val 'int'

        # ...but the original definition of 'struct Foo' is not in the scratch AST anymore
        # CHECK-NOT: FieldDecl {{.*}} m_val 'int'
