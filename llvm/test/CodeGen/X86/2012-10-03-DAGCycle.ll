; RUN: llc -mtriple=x86_64-apple-macosx -mcpu=corei7 < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.pluto.0 = type { %struct.bar.1, %struct.hoge.368* }
%struct.bar.1 = type { %i8* }
%i8 = type { i8 }
%struct.hoge.368 = type { i32, i32 }
%struct.widget.375 = type { i32, i32, %i8*, %struct.hoge.368* }

define fastcc void @bar(%struct.pluto.0* %arg) nounwind uwtable ssp align 2 {
bb:
  %tmp1 = alloca %struct.widget.375, align 8
  %tmp2 = getelementptr inbounds %struct.pluto.0* %arg, i64 0, i32 1
  %tmp3 = load %struct.hoge.368** %tmp2, align 8
  store %struct.pluto.0* %arg, %struct.pluto.0** undef, align 8
  %tmp = getelementptr inbounds %struct.widget.375* %tmp1, i64 0, i32 2
  %tmp4 = getelementptr %struct.pluto.0* %arg, i64 0, i32 0, i32 0
  %tmp5 = load %i8** %tmp4, align 8
  store %i8* %tmp5, %i8** %tmp, align 8
  %tmp6 = getelementptr inbounds %struct.widget.375* %tmp1, i64 0, i32 3
  store %struct.hoge.368* %tmp3, %struct.hoge.368** %tmp6, align 8
  br i1 undef, label %bb8, label %bb7

bb7:                                              ; preds = %bb
  unreachable

bb8:                                              ; preds = %bb
  unreachable
}
