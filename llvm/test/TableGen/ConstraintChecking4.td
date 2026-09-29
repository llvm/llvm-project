// RUN: not llvm-tblgen -gen-asm-writer -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "ConstraintChecking.inc"

// CHECK: [[FILE]]:[[@LINE+1]]:5: error: Output operands '$dest1' and '$dest2' of 'Foo' cannot be tied!
def Foo : TestInstructionWithConstraints<"$dest1 = $dest2">;
