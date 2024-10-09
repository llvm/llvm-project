// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s
// Verify that processors with same names result in an error.

include "llvm/Target/Target.td"

def MyTarget : Target;

def ProcessorB : ProcessorModel<"NameA", NoSchedModel, []>;

// CHECK: [[FILE]]:[[@LINE+2]]:5: error: Processor `NameA` is already defined.
// CHECK: [[FILE]]:[[@LINE-3]]:5: note: Previous definition here.
def ProcessorA : ProcessorModel<"NameA", NoSchedModel, []>;
