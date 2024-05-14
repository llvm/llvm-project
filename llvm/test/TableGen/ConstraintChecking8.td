// RUN: not llvm-tblgen -gen-asm-writer -DT0 -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s
// RUN: not llvm-tblgen -gen-asm-writer -DT1 -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK1
// RUN: not llvm-tblgen -gen-asm-writer -DT2 -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK2
// RUN: not llvm-tblgen -gen-asm-writer -DT3 -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK3
// RUN: not llvm-tblgen -gen-asm-writer -DT4 -I %p -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK4

include "ConstraintChecking.inc"

// Make sure exactly the token "=" appears.

#ifdef T0
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: Unrecognized constraint '$dest1 != $src2' in 'Foo'
def Foo : TestInstructionWithConstraints<"$dest1 != $src2">;
#endif

#ifdef T1
// CHECK1: [[FILE]]:[[@LINE+1]]:5: error: Unrecognized constraint '$dest1 == $src2' in 'Foo'
def Foo : TestInstructionWithConstraints<"$dest1 == $src2">;
#endif

#ifdef T2
// CHECK2: [[FILE]]:[[@LINE+1]]:5: error: Unrecognized constraint '= $rhs' in 'Foo'
def Foo : TestInstructionWithConstraints<"= $rhs">;
#endif

#ifdef T3
// CHECK3: [[FILE]]:[[@LINE+1]]:5: error: Unrecognized constraint '$lhs =' in 'Foo'
def Foo : TestInstructionWithConstraints<"$lhs =">;
#endif

#ifdef T4
// CHECK4: [[FILE]]:[[@LINE+1]]:5: error: Unrecognized constraint '=' in 'Foo'
def Foo : TestInstructionWithConstraints<"=">;
#endif
