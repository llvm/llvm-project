; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) #0
declare void @llvm.experimental.guard(i1, ...)
declare void @unknown_but_readonly() readonly

define void @test1(i8* %P, i8* %Q) {
  tail call void(i1,...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test1:

; CHECK:  Just Ref:  Ptr: i8* %P	<->  tail call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
; CHECK:  Just Ref:  Ptr: i8* %Q	<->  tail call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
; CHECK:  Both ModRef:  Ptr: i8* %P	<->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i1 false)
; CHECK:  Both ModRef:  Ptr: i8* %Q	<->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i1 false)
; CHECK:  Just Ref:   tail call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ] <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i1 false)
; CHECK:  Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i1 false) <->   tail call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
}

define void @test2() {
  tail call void(i1,...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
  tail call void @unknown_but_readonly()
  ret void
; CHECK-LABEL: Function: test2:
; CHECK:  NoModRef:   tail call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ] <->   tail call void @unknown_but_readonly()
; CHECK:  NoModRef:   tail call void @unknown_but_readonly() <->   tail call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
}
