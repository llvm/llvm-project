// RUN: not llvm-tblgen -gen-asm-writer -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "ConstraintChecking.inc"

// CHECK: [[FILE]]:[[@LINE+1]]:1: error: Operand '$src1' of 'Foo' cannot have multiple constraints!
def Foo : TestInstructionWithConstraints<"$dest1 = $src1, $dest2 = $src1">;
