; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S < %s | FileCheck %s
; RUN: opt -strip-debug -codegenprepare -S < %s | FileCheck %s
; REQUIRES: x86-registered-target

; In its current state, CodeGenPrepare should not optimize empty blocks after a switch
; (See CodeGenPrepare::isMergingEmptyBlockProfitable)
; This should also be the case when they contain debug information. (sw.bb block)
; Check this by verifying that the switch labels remain the same

; CHECK: while.cond:
; CHECK-NEXT:  switch i32 undef, label %sw.default [
; CHECK-NEXT:    i32 45, label %sw.bb
; CHECK-NEXT:    i32 104, label %while.cond.lbl_crit_edge
; CHECK-NEXT:    i32 122, label %while.cond.lbl_crit_edge
; CHECK-NEXT:    i32 115, label %while.cond134.preheader
; CHECK-NEXT:    i32 100, label %sw.bb29
; CHECK-NEXT:    i32 105, label %sw.bb29
; CHECK-NEXT:  ]

; ModuleID = 'bugpoint-reduced-instructions.bc'
source_filename = "foobar.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; Function Attrs: noinline nounwind uwtable
define dso_local void @foobar(ptr nocapture %arg0) local_unnamed_addr #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %entry
  switch i32 undef, label %sw.default [
    i32 45, label %sw.bb
    i32 104, label %while.cond.lbl_crit_edge
    i32 122, label %while.cond.lbl_crit_edge
    i32 115, label %while.cond134.preheader
    i32 100, label %sw.bb29
    i32 105, label %sw.bb29
  ]

while.cond.lbl_crit_edge:                         ; preds = %while.cond, %while.cond
  br label %lbl

while.cond134.preheader:                          ; preds = %while.cond
  unreachable

sw.bb:                                            ; preds = %while.cond
  call void @llvm.dbg.value(metadata i32 3, metadata !11, metadata !DIExpression()), !dbg !15
  br label %lbl

sw.bb29:                                          ; preds = %while.cond, %while.cond
  unreachable

sw.default:                                       ; preds = %while.cond
  unreachable

lbl:                                              ; preds = %sw.bb, %while.cond.lbl_crit_edge
  %var03.1 = phi i32 [ 0, %while.cond.lbl_crit_edge ], [ 3, %sw.bb ]
  br label %phi_use

phi_use:
  store i32 %var03.1, ptr %arg0
  unreachable
}

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version XXX ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !2)
!1 = !DIFile(filename: "foobar.c", directory: "./")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DISubroutineType(types: !7)
!7 = !{!4, !4, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !DILocalVariable(name: "var03", scope: !12, file: !1, line: 3, type: !4)
!12 = distinct !DISubprogram(name: "foobar", scope: !1, file: !1, line: 2, type: !13, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !5, !8}
!15 = !DILocation(line: 4, column: 16, scope: !12)
