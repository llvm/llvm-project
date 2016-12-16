; RUN: llvm-as -o %t.bc %s
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so -plugin-opt=emit-llvm \
; RUN:    --no-map-whole-files -r -o %t2.bc %t.bc
; RUN: llvm-dis < %t2.bc -o - | FileCheck %s

; CHECK: main

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @main() {
  ret i32 0
}
