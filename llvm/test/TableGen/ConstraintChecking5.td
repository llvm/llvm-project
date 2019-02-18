// RUN: not llvm-tblgen -gen-asm-writer -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "ConstraintChecking.inc"

// CHECK: [[FILE]]:[[@LINE+1]]:1: error: Input operands '$src1' and '$src2' of 'Foo' cannot be tied!
def Foo : TestInstructionWithConstraints<"$src1 = $src2">;
