"""
Test that re-running a process from within the same target
after rebuilding the a dynamic library flushes the scratch
TypeSystems tied to that process.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestRerun(TestBase):
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

        # Build a.out
        self.build(dictionary={'EXE':'a.out',
                               'CXX_SOURCES':'main.cpp'})

        # Build libfoo.dylib
        self.build(dictionary={'DYLIB_CXX_SOURCES':'lib.cpp',
                               'DYLIB_ONLY':'YES',
                               'DYLIB_NAME':'foo',
                               'USE_LIBDL':'1',
                               'LD_EXTRAS':'-L.'})

        (target, _, _, bkpt) = \
                lldbutil.run_to_source_breakpoint(self, 'return', lldb.SBFileSpec('main.cpp'))

        self.expect_expr('*foo', result_type='Foo', result_children=[
                ValueCheck(name='m_val', value='42')
            ])

        # Re-build libfoo.dylib
        self.build(dictionary={'DYLIB_CXX_SOURCES':'rebuild.cpp',
                               'DYLIB_ONLY':'YES',
                               'DYLIB_NAME':'foo',
                               'USE_LIBDL':'1',
                               'LD_EXTRAS':'-L.'})

        self.runCmd('process launch')
        (target, _, _, bkpt) = \
                lldbutil.run_to_source_breakpoint(self, 'return', lldb.SBFileSpec('main.cpp'))

        self.expect_expr('*foo', result_type='Foo', result_children=[
            ValueCheck(name='Base', children=[
                ValueCheck(name='m_base_val', value='42')
            ]),
            ValueCheck(name='m_derived_val', value='137')
        ])

        self.filecheck("target module dump ast", __file__)

        # The new definition 'struct Foo' is in the scratch AST
        # CHECK:      |-CXXRecordDecl {{.*}} struct Foo definition
        # CHECK:      | |-public 'Base'
        # CHECK-NEXT: | `-FieldDecl {{.*}} m_derived_val 'int'
        # CHECK-NEXT: `-CXXRecordDecl {{.*}} struct Base definition

        # ...but the original definition of 'struct Foo' is not in the scratch AST anymore
        # CHECK-NOT: FieldDecl {{.*}} m_val 'int'
