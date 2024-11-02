; RUN: llc -mtriple=x86_64-apple-macosx -mcpu=corei7 < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.pluto.0 = type { %struct.bar.1, ptr }
%struct.bar.1 = type { ptr }
%i8 = type { i8 }
%struct.hoge.368 = type { i32, i32 }
%struct.widget.375 = type { i32, i32, ptr, ptr }

define fastcc void @bar(ptr %arg) nounwind uwtable ssp align 2 {
bb:
  %tmp1 = alloca %struct.widget.375, align 8
  %tmp2 = getelementptr inbounds %struct.pluto.0, ptr %arg, i64 0, i32 1
  %tmp3 = load ptr, ptr %tmp2, align 8
  store ptr %arg, ptr undef, align 8
  %tmp = getelementptr inbounds %struct.widget.375, ptr %tmp1, i64 0, i32 2
  %tmp5 = load ptr, ptr %arg, align 8
  store ptr %tmp5, ptr %tmp, align 8
  %tmp6 = getelementptr inbounds %struct.widget.375, ptr %tmp1, i64 0, i32 3
  store ptr %tmp3, ptr %tmp6, align 8
  br i1 undef, label %bb8, label %bb7

bb7:                                              ; preds = %bb
  unreachable

bb8:                                              ; preds = %bb
  unreachable
}
