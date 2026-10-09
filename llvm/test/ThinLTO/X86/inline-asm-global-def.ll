; A global symbol defined by inline asm in the body of a function has to be
; known to the linker, and the function has to be kept although nothing in the
; IR refers to it.

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/inline-asm-global-def.ll -o %t2.bc
; RUN: llvm-lto2 run %t.bc %t2.bc -o %t.out \
; RUN:   -r=%t.bc,carrier,px -r=%t.bc,object_in_asm,px -r=%t.bc,object_in_c,px -r=%t.bc,object_in_c, \
; RUN:   -r=%t2.bc,use,px -r=%t2.bc,object_in_asm, -r=%t2.bc,object_in_c,
; RUN: llvm-nm %t.out.1 %t.out.2 | FileCheck %s

; The same with regular LTO.
; RUN: opt %s -o %t3.bc
; RUN: opt %p/Inputs/inline-asm-global-def.ll -o %t4.bc
; RUN: llvm-lto2 run %t3.bc %t4.bc -o %t5.out \
; RUN:   -r=%t3.bc,carrier,px -r=%t3.bc,object_in_asm,px -r=%t3.bc,object_in_c,px -r=%t3.bc,object_in_c, \
; RUN:   -r=%t4.bc,use,px -r=%t4.bc,object_in_asm, -r=%t4.bc,object_in_c,
; RUN: llvm-nm %t5.out.0 | FileCheck %s

; CHECK-DAG: T carrier
; CHECK-DAG: T object_in_asm
; CHECK-DAG: {{[DR]}} object_in_c

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@object_in_c = dso_local constant i32 5

define dso_local void @carrier() {
  call void asm sideeffect ".globl object_in_asm\0Aobject_in_asm:\0A.quad object_in_c\0A", ""()
  ret void
}
