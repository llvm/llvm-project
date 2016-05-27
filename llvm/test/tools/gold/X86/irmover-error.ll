; RUN: llvm-as -o %t1.bc %s
; RUN: llvm-as -o %t2.bc %S/Inputs/irmover-error.ll
; RUN: not %gold -plugin %llvmshlibdir/LLVMgold.so -o %t %t1.bc %t2.bc 2>&1 | FileCheck %s

; CHECK: fatal error: Failed to link module {{.*}}2.bc: linking module flags 'foo': IDs have conflicting values

!0 = !{ i32 1, !"foo", i32 1 }

!llvm.module.flags = !{ !0 }
