; RUN: llc -mcpu=corei7 -no-stack-coloring=false < %s

; Make sure that we don't crash when dbg values are used.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define void @foo() nounwind uwtable ssp {
entry:
  %x.i = alloca i8, align 1
  %y.i = alloca [256 x i8], align 16
  %0 = getelementptr inbounds [256 x i8]* %y.i, i64 0, i64 0
  br label %for.body

for.body:
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.start(i64 -1, i8* %x.i) nounwind
  call void @llvm.dbg.declare(metadata !{i8* %x.i}, metadata !22) nounwind
  br label %for.body
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}
!0 = metadata !{i32 524305, metadata !1, i32 1, metadata !"clang", i1 true, metadata !"", i32 0, metadata !2, metadata !2, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"t.c", metadata !""}
!16 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6}
!2 = metadata !{i32 0}
!22 = metadata !{i32 786688, null, metadata !"x", metadata !2, i32 16, metadata !16, i32 0, i32 0}
!23 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
