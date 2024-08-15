; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_


; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Based on optimized IR from C source:
;; int main () {
;;   char a1[__INT_MAX__];
;;   a1[__INT_MAX__ - 1] = 5;
;;   return a1[__INT_MAX__ - 1];
;; }
;;
;; Check extremely large types don't cause a crash.
; CHECK: DBG_VALUE 5, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 4294967280, 8)
; CHECK: DBG_VALUE 6, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 8)
; CHECK: DBG_VALUE 7, $noreg, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 8)

define dso_local i32 @main() local_unnamed_addr !dbg !10 {
entry:
;; FIXME: SROA currently creates incorrect fragments if bit_offset > max(u32),
;; with and without assignment-tracking.
  tail call void @llvm.dbg.value(metadata i8 5, metadata !15, metadata !DIExpression(DW_OP_LLVM_fragment, 4294967280, 8)), !dbg !20
;; These two were inserted by hand.
  tail call void @llvm.dbg.value(metadata i8 6, metadata !22, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 8)), !dbg !20
  tail call void @llvm.dbg.value(metadata i8 7, metadata !23, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 8)), !dbg !20
  ret i32 5, !dbg !21
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 18.0.0"}
!10 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !11, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "a1", scope: !10, file: !1, line: 5, type: !16)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 17179869176, elements: !18)
!17 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!18 = !{!19}
!19 = !DISubrange(count: 2147483647)
!20 = !DILocation(line: 0, scope: !10)
!21 = !DILocation(line: 7, column: 3, scope: !10)
!22 = !DILocalVariable(name: "a2", scope: !10, file: !1, line: 5, type: !16)
!23 = !DILocalVariable(name: "a3", scope: !10, file: !1, line: 5, type: !16)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 4294967232, elements: !18)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 4294967233, elements: !18)
