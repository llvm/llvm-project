target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc -verify-machineinstrs < %s | FileCheck %s

define i64 @test1() nounwind {
entry:
  %r = call i64 @llvm.readcyclecounter()
  ret i64 %r
}

; CHECK: @test1
; CHECK: mfspr 3, 268

declare i64 @llvm.readcyclecounter()

