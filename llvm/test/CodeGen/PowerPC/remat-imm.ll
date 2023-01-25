; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; ModuleID = 'test.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-linux"

@.str = private unnamed_addr constant [6 x i8] c"%d,%d\00", align 1

define i32 @main() nounwind {
entry:
; CHECK: li 4, 128
; CHECK-NOT: mr 4, {{.*}}
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i32 128, i32 128) nounwind
  ret i32 0
}

declare i32 @printf(ptr nocapture, ...) nounwind
