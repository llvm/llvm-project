; RUN: opt -S -passes=div-rem-pairs -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; Check that DivRemPairs's optimizeDivRem() propagates the debug location of
; replaced `%rem` to the new srem or urem instruction.
; @decompose_illegal_srem_same_block tests it when the operands of `%rem` is signed.
; @decompose_illegal_urem_same_block tests it when the operands of `%rem` is unsigned.

define void @decompose_illegal_srem_same_block(i32 %a, i32 %b) !dbg !5 {
; CHECK-LABEL: define void @decompose_illegal_srem_same_block(
; CHECK:         %rem.recomposed = srem i32 [[A:%.*]], [[B:%.*]], !dbg [[DBG10:![0-9]+]]
  %div = sdiv i32 %a, %b, !dbg !8
  %t0 = mul i32 %div, %b, !dbg !9
  %rem = sub i32 %a, %t0, !dbg !10
  call void @foo(i32 %rem, i32 %div), !dbg !11
  ret void, !dbg !12
}

define void @decompose_illegal_urem_same_block(i32 %a, i32 %b) !dbg !13 {
; CHECK-LABEL: define void @decompose_illegal_urem_same_block(
; CHECK:         %rem.recomposed = urem i32 [[A:%.*]], [[B:%.*]], !dbg [[DBG16:![0-9]+]]
  %div = udiv i32 %a, %b, !dbg !14
  %t0 = mul i32 %div, %b, !dbg !15
  %rem = sub i32 %a, %t0, !dbg !16
  call void @foo(i32 %rem, i32 %div), !dbg !17
  ret void, !dbg !18
}

declare void @foo(i32, i32)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "realrem_preverse.ll", directory: "/")
!2 = !{i32 10}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "decompose_illegal_srem_same_block", linkageName: "decompose_illegal_srem_same_block", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = distinct !DISubprogram(name: "decompose_illegal_urem_same_block", linkageName: "decompose_illegal_urem_same_block", scope: null, file: !1, line: 6, type: !6, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DILocation(line: 6, column: 1, scope: !13)
!15 = !DILocation(line: 7, column: 1, scope: !13)
!16 = !DILocation(line: 8, column: 1, scope: !13)
!17 = !DILocation(line: 9, column: 1, scope: !13)
!18 = !DILocation(line: 10, column: 1, scope: !13)

; CHECK: [[DBG10]] = !DILocation(line: 3,
; CHECK: [[DBG16]] = !DILocation(line: 8,

