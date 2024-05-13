; RUN: opt -S -passes=gvn < %s | FileCheck %s
; Check that loads from calloc are recognized as being zero.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define i32 @test1() {
  %1 = tail call noalias ptr @calloc(i64 1, i64 4)
  ; This load is trivially constant zero
  %2 = load i32, ptr %1, align 4
  ret i32 %2

; CHECK-LABEL: @test1(
; CHECK-NOT: %2 = load i32, ptr %1, align 4
; CHECK: ret i32 0

; CHECK_NO_LIBCALLS-LABEL: @test1(
; CHECK_NO_LIBCALLS: load
; CHECK_NO_LIBCALLS: ret i32 %

}

define i32 @as_invoke(i1 %c) personality ptr undef {
bb3:
  %mem = invoke noalias ptr @calloc(i64 1, i64 4)
  to label %bb4 unwind label %bb1

bb1:
  %lp = landingpad { ptr, i32 } cleanup
  ret i32 0

bb4:
  ; This load is trivially constant zero
  %res = load i32, ptr %mem, align 4
  ret i32 %res

; CHECK-LABEL: @as_invoke(
; CHECK-NOT: %3 = load i32, ptr %2, align 4
; CHECK: ret i32 0

; CHECK_NO_LIBCALLS-LABEL: @as_invoke(
; CHECK_NO_LIBCALLS: load
; CHECK_NO_LIBCALLS: ret i32 %
}

declare noalias ptr @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1)
