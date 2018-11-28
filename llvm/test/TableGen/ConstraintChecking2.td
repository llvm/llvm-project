// RUN: not llvm-tblgen -gen-asm-writer -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "ConstraintChecking.inc"

// CHECK: [[FILE]]:[[@LINE+1]]:1: error: Illegal format for @earlyclobber constraint in 'Foo'
def Foo : TestInstructionWithConstraints<"@earlyclobber ">;
