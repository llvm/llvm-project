; RUN: opt < %s -memcpyopt -S | FileCheck %s
; Test memcpy-memcpy dependencies across invoke edges.

; Test that memcpyopt works across the non-unwind edge of an invoke.

define hidden void @test_normal(i8* noalias %dst, i8* %src) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %temp = alloca i8, i32 64
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %temp, i8* nonnull %src, i64 64, i32 8, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %temp, i8* nonnull %src, i64 64, i32 8, i1 false)
  invoke void @invoke_me()
          to label %try.cont unwind label %lpad

lpad:
  landingpad { i8*, i32 }
          catch i8* null
  ret void

try.cont:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %temp, i64 64, i32 8, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 64, i32 8, i1 false)
  ret void
}

; Test that memcpyopt works across the unwind edge of an invoke.

define hidden void @test_unwind(i8* noalias %dst, i8* %src) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %temp = alloca i8, i32 64
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %temp, i8* nonnull %src, i64 64, i32 8, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %temp, i8* nonnull %src, i64 64, i32 8, i1 false)
  invoke void @invoke_me()
          to label %try.cont unwind label %lpad

lpad:
  landingpad { i8*, i32 }
          catch i8* null
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %temp, i64 64, i32 8, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 64, i32 8, i1 false)
  ret void

try.cont:
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)
declare i32 @__gxx_personality_v0(...)
declare void @invoke_me() readnone
