; RUN: opt -adce %s -S -o - | FileCheck %s
; RUN: opt -bdce %s -S -o - | FileCheck %s
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"
define void @f(i32) !dbg !8 {
entry:
  %p_x = inttoptr i32 %0 to ptr
  %i_x = ptrtoint ptr %p_x to i32
  ; CHECK: call void @llvm.dbg.value(metadata i32 %0,
  ; CHECK-SAME: !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_unsigned,
  ; CHECK-SAME:               DW_OP_LLVM_convert, 64, DW_ATE_unsigned,
  ; CHECK-SAME:               DW_OP_LLVM_convert, 64, DW_ATE_unsigned,
  ; CHECK-SAME:               DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value))
  call void @llvm.dbg.value(metadata i32 %i_x, metadata !11, metadata !DIExpression()), !dbg !13
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
