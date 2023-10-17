; RUN: opt -S %s -passes=loop-deletion | FileCheck %s

;; static int foo(int Param) __attribute__((always_inline));
;; static int foo(int Param) { return Param * Param * 2; }
;;
;; static int zero() __attribute__((always_inline));
;; static int zero() { return 0; }
;;
;; void test() {
;;   int Constant = 0;
;;   for (int Index = zero(); Index < 0; ++Index) {
;;     foo(Index);
;;     Constant = 5;
;;     foo(Index + 1);
;;   }
;; }
;;
;; Check that a loop invariant value in a dbg.value inside the dead
;; loop is preserved.

; CHECK-LABEL: for.end:
; CHECK-NEXT:    @llvm.dbg.value({{.+}} undef, metadata ![[VAR1:[0-9]+]],{{.+}}), !dbg ![[DBG1:[0-9]+]]
; CHECK-NEXT:    @llvm.dbg.value({{.+}} 5, metadata ![[VAR2:[0-9]+]],{{.+}}), !dbg ![[DBG2:[0-9]+]]

; CHECK-DAG: ![[VAR1]] = !DILocalVariable(name: "Index"
; CHECK-DAG: ![[VAR2]] = !DILocalVariable(name: "Constant"

; CHECK-DAG: ![[DBG1]] = !DILocation(line: 0
; CHECK-DAG: ![[DBG2]] = !DILocation(line: 0

define dso_local void @test() !dbg !9 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !18
  br label %for.end, !dbg !19

for.end:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i32 undef, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 5, metadata !13, metadata !DIExpression()), !dbg !15
  ret void, !dbg !30
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test", directory: "", checksumkind: CSK_MD5, checksum: "53f7b620ca60e4fc7af4e6e4cdac8eea")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 1, !"MaxTLSAlign", i32 65536}
!8 = !{!"clang version 17.0.0"}
!9 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 7, type: !10, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{}
!13 = !DILocalVariable(name: "Constant", scope: !9, file: !1, line: 8, type: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 0, scope: !9)
!16 = !DILocalVariable(name: "Index", scope: !17, file: !1, line: 9, type: !14)
!17 = distinct !DILexicalBlock(scope: !9, file: !1, line: 9)
!18 = !DILocation(line: 0, scope: !17)
!19 = !DILocation(line: 9, scope: !17)
!21 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !22, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !12)
!22 = !DISubroutineType(types: !23)
!23 = !{!14, !14}
!24 = !DILocation(line: 0, scope: !21, inlinedAt: !25)
!25 = distinct !DILocation(line: 10, scope: !26)
!26 = distinct !DILexicalBlock(scope: !27, file: !1, line: 9)
!27 = distinct !DILexicalBlock(scope: !17, file: !1, line: 9)
!28 = !DILocation(line: 0, scope: !21, inlinedAt: !29)
!29 = distinct !DILocation(line: 12, scope: !26)
!30 = !DILocation(line: 14, scope: !9)
