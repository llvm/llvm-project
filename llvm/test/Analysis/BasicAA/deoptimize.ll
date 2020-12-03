; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) #0
declare void @llvm.experimental.deoptimize.void(...)
declare void @unknown_but_readonly() readonly

define void @test1(i8* %p) {
  call void(...) @llvm.experimental.deoptimize.void() [ "deopt"() ]
  ret void

; CHECK-LABEL: Function: test1:
; CHECK:  Just Ref: Ptr: i8* %p <-> call void (...) @llvm.experimental.deoptimize.isVoid() [ "deopt"() ]
}
