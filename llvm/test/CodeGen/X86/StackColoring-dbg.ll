; RUN: llc -mcpu=corei7 -no-stack-coloring=false < %s

; Make sure that we don't crash when dbg values are used.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @foo() nounwind uwtable ssp {
entry:
  %x.i = alloca i8, align 1
  %y.i = alloca [256 x i8], align 16
  br label %for.body

for.body:
  call void @llvm.lifetime.end.p0(i64 -1, ptr %y.i) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %x.i) nounwind
  call void @llvm.dbg.declare(metadata ptr %x.i, metadata !22, metadata !DIExpression()) nounwind, !dbg !DILocation(scope: !2)
  br label %for.body
}


declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}
!0 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "clang", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !{}, retainedTypes: !{})
!1 = !DIFile(filename: "t.c", directory: "")
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!2 = distinct !DISubprogram(unit: !0)
!22 = !DILocalVariable(name: "x", line: 16, scope: !2, file: !1, type: !16)
!23 = !{i32 1, !"Debug Info Version", i32 3}
