; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Ensure that the order of several debug intrinsics between non-debug
;; instructions is maintained.

; CHECK-DAG: ![[A:[0-9]+]] = !DILocalVariable(name: "a",
; CHECK-DAG: ![[B:[0-9]+]] = !DILocalVariable(name: "b",
; CHECK-DAG: ![[C:[0-9]+]] = !DILocalVariable(name: "c",

; CHECK:      DBG_VALUE $esi, $noreg, ![[B]], !DIExpression()
; CHECK-NEXT: DBG_VALUE $edx, $noreg, ![[C]], !DIExpression()
; CHECK-NEXT: DBG_VALUE $esi, $noreg, ![[A]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE $edx, $noreg, ![[A]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)

target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @fun(i64 %a.coerce, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.assign(metadata i32 %b, metadata !17, metadata !DIExpression(), metadata !19, metadata ptr undef, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.assign(metadata i32 %c, metadata !18, metadata !DIExpression(), metadata !21, metadata ptr undef, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.assign(metadata i32 %b, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !22, metadata ptr undef, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.assign(metadata i32 %c, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !23, metadata ptr undef, metadata !DIExpression()), !dbg !20
  %mul = mul nsw i32 %c, %b, !dbg !24
  ret i32 %mul, !dbg !25
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 2, size: 64, elements: !12)
!12 = !{!13, !14}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !1, line: 2, baseType: !10, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !11, file: !1, line: 2, baseType: !10, size: 32, offset: 32)
!15 = !{!16, !17, !18}
!16 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 3, type: !11)
!17 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !1, line: 3, type: !10)
!18 = !DILocalVariable(name: "c", arg: 3, scope: !7, file: !1, line: 3, type: !10)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 0, scope: !7)
!21 = distinct !DIAssignID()
!22 = distinct !DIAssignID()
!23 = distinct !DIAssignID()
!24 = !DILocation(line: 6, column: 14, scope: !7)
!25 = !DILocation(line: 6, column: 3, scope: !7)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
