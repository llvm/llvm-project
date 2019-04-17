; RUN: opt -S -basicaa -gvn < %s | FileCheck %s
; RUN: opt -S -basicaa -gvn -disable-simplify-libcalls < %s | FileCheck %s -check-prefix=CHECK_NO_LIBCALLS
; Check that loads from calloc are recognized as being zero.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define i32 @test1() {
  %1 = tail call noalias i8* @calloc(i64 1, i64 4)
  %2 = bitcast i8* %1 to i32*
  ; This load is trivially constant zero
  %3 = load i32, i32* %2, align 4
  ret i32 %3

; CHECK-LABEL: @test1(
; CHECK-NOT: %3 = load i32, i32* %2, align 4
; CHECK: ret i32 0

; CHECK_NO_LIBCALLS-LABEL: @test1(
; CHECK_NO_LIBCALLS: load
; CHECK_NO_LIBCALLS: ret i32 %

}

declare noalias i8* @calloc(i64, i64)
