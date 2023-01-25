// RUN: not llvm-tblgen -gen-asm-writer -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "ConstraintChecking.inc"

// (This is illegal because the '=' has to be surrounded by whitespace)

// CHECK: [[FILE]]:[[@LINE+1]]:5: error: Illegal format for tied-to constraint in 'Foo'
def Foo : TestInstructionWithConstraints<"$dest1=$dest2">;
