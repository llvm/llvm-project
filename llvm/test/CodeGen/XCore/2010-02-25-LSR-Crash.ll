; RUN: llc < %s -mtriple=xcore
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "xcore-xmos-elf"

%0 = type { i32 }
%struct.dwarf_fde = type <{ i32, i32, [0 x i8] }>
%struct.object = type { ptr, ptr, ptr, %union.anon, %0, ptr }
%union.anon = type { ptr }

define ptr @search_object(ptr %ob, ptr %pc) {
entry:
  br i1 undef, label %bb3.i15.i.i, label %bb2

bb3.i15.i.i:                                      ; preds = %bb3.i15.i.i, %entry
  %indvar.i.i.i = phi i32 [ %indvar.next.i.i.i, %bb3.i15.i.i ], [ 0, %entry ] ; <i32> [#uses=2]
  %tmp137 = sub i32 0, %indvar.i.i.i              ; <i32> [#uses=1]
  %scevgep13.i.i.i = getelementptr i32, ptr undef, i32 %tmp137 ; <ptr> [#uses=2]
  %0 = load ptr, ptr %scevgep13.i.i.i, align 4 ; <ptr> [#uses=0]
  store i32 undef, ptr %scevgep13.i.i.i
  %indvar.next.i.i.i = add i32 %indvar.i.i.i, 1   ; <i32> [#uses=1]
  br label %bb3.i15.i.i

bb2:                                              ; preds = %entry
  ret ptr undef
}
