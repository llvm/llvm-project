; RUN: opt < %s -licm -S | FileCheck %s
; RUN: opt < %s -strip-debug -licm -S | FileCheck %s
; RUN: opt < %s -licm -enable-mssa-loop-dependency=true -verify-memoryssa -S | FileCheck %s --check-prefixes=CHECK,MSSA

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Verify that the sdiv is hoisted out of the loop
; even in the presence of a preceding debug intrinsic.

@a = global i32 0
@b = global i32 0
@c = global i32 0

define void @fn1() !dbg !6 {
; CHECK-LABEL: @fn1(
; CHECK-NEXT: [[_TMP2:%.*]] = load i32, i32* @a, align 4
; CHECK-NEXT: [[_TMP3:%.*]] = load i32, i32* @b, align 4
; CHECK-NEXT: [[_TMP4:%.*]] = sdiv i32 [[_TMP2]], [[_TMP3]]
; MSSA-NEXT: store i32 [[_TMP4:%.*]], i32* @c, align 4
; CHECK-NEXT: br label [[BB3:%.*]]

  br label %bb3

bb3:                                              ; preds = %bb3, %0
  call void @llvm.dbg.value(metadata i32* @c, metadata !10, metadata !DIExpression(DW_OP_deref)), !dbg !12
  %_tmp2 = load i32, i32* @a, align 4
  %_tmp3 = load i32, i32* @b, align 4
  %_tmp4 = sdiv i32 %_tmp2, %_tmp3
  store i32 %_tmp4, i32* @c, align 4
  %_tmp6 = load volatile i32, i32* @c, align 4
  br label %bb3
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "foo", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"foo"}
!6 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "f", scope: !11, line: 5, type: !9)
!11 = distinct !DILexicalBlock(scope: !6, file: !1, line: 4, column: 12)
!12 = !DILocation(line: 5, column: 9, scope: !11)
