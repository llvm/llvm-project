"""
Test that re-running a process from within the same target
after rebuilding the executable flushes the scratch TypeSystems
tied to that process.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *


class TestRerunExpr(TestBase):
    # FIXME: on Windows rebuilding the binary isn't enough to unload it
    #        on progrem restart. One will have to try hard to evict
    #        the module from the ModuleList (possibly including a call to
    #        SBDebugger::MemoryPressureDetected.
    @skipIfWindows
    def test(self):
        """
        Tests whether re-launching a process without destroying
        the owning target keeps invalid ASTContexts in the
        scratch AST's importer.

        We test this by:
        1. Evaluating an expression to import 'struct Foo' into
           the scratch AST
        2. Change the definition of 'struct Foo' and rebuild the executable
        3. Re-launch the process
        4. Evaluate the same expression in (1). We expect to have only
           the latest definition of 'struct Foo' in the scratch AST.
        """
        self.build(dictionary={"CXX_SOURCES": "main.cpp", "EXE": "a.out"})

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        target.BreakpointCreateBySourceRegex("return", lldb.SBFileSpec("rebuild.cpp"))
        target.BreakpointCreateBySourceRegex("return", lldb.SBFileSpec("main.cpp"))
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.expect_expr(
            "foo",
            result_type="Foo",
            result_children=[ValueCheck(name="m_val", value="42")],
        )

        # Delete the executable to force make to rebuild it.
        remove_file(exe)
        self.build(dictionary={"CXX_SOURCES": "rebuild.cpp", "EXE": "a.out"})

        # Rerun program within the same target
        process.Destroy()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.expect_expr(
            "foo",
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
