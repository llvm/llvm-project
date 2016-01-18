; RUN: llvm-as %s -o %t.o
; REQUIRES: asserts

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=disable-verify \
; RUN:    --plugin-opt=-debug-pass=Arguments \
; RUN:    -shared %t.o -o %t2.o 2>&1 | FileCheck %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=-debug-pass=Arguments \
; RUN:    -shared %t.o -o %t2.o 2>&1 | FileCheck %s -check-prefix=VERIFY

target triple = "x86_64-unknown-linux-gnu"

; -disable-verify should disable output verification from the optimization
; pipeline.
; CHECK: Pass Arguments: {{.*}} -verify -ipsccp
; CHECK-NOT: -verify

; VERIFY: Pass Arguments: {{.*}} -verify {{.*}} -verify

define void @f() {
entry:
  ret void
}
