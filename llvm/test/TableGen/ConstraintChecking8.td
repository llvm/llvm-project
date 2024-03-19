// RUN: not llvm-tblgen -gen-asm-writer -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "ConstraintChecking.inc"

// Make sure exactly the token "=" appears.

// CHECK: [[FILE]]:[[@LINE+1]]:5: error: Unrecognized constraint '$dest1 != $src2' in 'Foo'
def Foo : TestInstructionWithConstraints<"$dest1 != $src2">;
