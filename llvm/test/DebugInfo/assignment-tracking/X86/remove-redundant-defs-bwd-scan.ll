; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Hand-written to test assignment tracking analysis' removal of redundant
;; debug loc definitions. Checks written inline.

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @f() #0 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !15
  call void @llvm.dbg.value(metadata i64 1, metadata !14, metadata !DIExpression()), !dbg !15
;; def [0 -> 32)
;; def [0          -> 64)
;; Second frag fully contains the first so first should be removed.
; CHECK:      DBG_VALUE 1,
  call void @step()
  call void @llvm.dbg.value(metadata i32 2, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !15
  call void @llvm.dbg.value(metadata i32 3, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !15
;; def [0 -> 32)
;; def         [32 -> 64)
;; Frags don't overlap so no removal should take place:
; CHECK:      DBG_VALUE 2,
; CHECK-NEXT: DBG_VALUE 3,
  call void @step()
  call void @llvm.dbg.value(metadata i16 4, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 16)), !dbg !15
  call void @llvm.dbg.value(metadata i8 5, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 8, 8)), !dbg !15
;; def [0      -> 16)
;; def      [8 -> 16)
;; Second frag doesn't fully contain first so no removal should take place:
; CHECK:      DBG_VALUE 4,
; CHECK-NEXT: DBG_VALUE 5,
  call void @step()
  call void @llvm.dbg.value(metadata i16 6, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 16)), !dbg !15
  call void @llvm.dbg.value(metadata i8 7, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 8, 8)), !dbg !15
  call void @llvm.dbg.value(metadata i32 8, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 8, 16)), !dbg !15
;; def [0      -> 16)
;; def      [8 -> 16)
;; def      [8       -> 24)
;; Middle frag is fully contained by the last so should be removed.
; CHECK:      DBG_VALUE 6,
; CHECK-NEXT: DBG_VALUE 8,
  ret void
}

declare void @step()
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !1000}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 14.0.0"}
!8 = distinct !DISubprogram(name: "f", linkageName: "_Z1fl", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!12 = !{}
!13 = distinct !DIAssignID()
!14 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!15 = !DILocation(line: 0, scope: !8)
!16 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
