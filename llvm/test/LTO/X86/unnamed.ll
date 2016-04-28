; RUN: llvm-as -o %t.bc %s
; RUN: llvm-lto -save-merged-module -o %t2 %t.bc
; RUN: llvm-dis -o - %t2.merged.bc | FileCheck %s

; CHECK-NOT: global i32

target triple = "x86_64-unknown-linux-gnu"

@0 = private global i32 42
@foo = constant i32* @0
