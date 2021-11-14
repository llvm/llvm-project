; RUN: opt -mtriple arm64e-apple-darwin -adce %s -S -o - | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

declare i64 @llvm.ptrauth.auth(i64, i32, i64)

define void @f(i64 %arg, i64 %arg1) !dbg !8 {
entry:
  %tmp = call i64 @llvm.ptrauth.auth(i64 %arg, i32 0, i64 %arg1)
  ; CHECK: call void @llvm.dbg.value(metadata i64 %arg,
  ; CHECK-SAME: !DIExpression())
  call void @llvm.dbg.value(metadata i64 %tmp, metadata !11, metadata !DIExpression()), !dbg !13
  ret void, !dbg !13
}
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "salavage.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 1, column: 1, scope: !8)
