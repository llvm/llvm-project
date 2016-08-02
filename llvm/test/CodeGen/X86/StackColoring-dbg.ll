; RUN: llc -mcpu=corei7 -no-stack-coloring=false < %s

; Make sure that we don't crash when dbg values are used.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @foo() nounwind uwtable ssp {
entry:
  %x.i = alloca i8, align 1
  %y.i = alloca [256 x i8], align 16
  %0 = getelementptr inbounds [256 x i8], [256 x i8]* %y.i, i64 0, i64 0
  br label %for.body

for.body:
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.start(i64 -1, i8* %x.i) nounwind
  call void @llvm.dbg.declare(metadata i8* %x.i, metadata !22, metadata !DIExpression()) nounwind, !dbg !DILocation(scope: !2)
  br label %for.body
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}
!0 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "clang", isOptimized: true, emissionKind: 0, file: !1, enums: !{}, retainedTypes: !{})
!1 = !DIFile(filename: "t.c", directory: "")
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!2 = distinct !DISubprogram()
!22 = !DILocalVariable(name: "x", line: 16, scope: !2, file: !1, type: !16)
!23 = !{i32 1, !"Debug Info Version", i32 3}
