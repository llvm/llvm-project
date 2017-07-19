; This test checks if debug loc is propagated to load/store created by GVN/Instcombine.
; RUN: opt < %s -gvn -S | FileCheck %s --check-prefixes=ALL,GVN
; RUN: opt < %s -gvn -instcombine -S | FileCheck %s --check-prefixes=ALL,INSTCOMBINE

; struct node {
;  int  *v;
; struct desc *descs;
; };

; struct desc {
;  struct node *node;
; };

; extern int bar(void *v, void* n);

; int test(struct desc *desc)
; {
;  void *v, *n;
;  v = !desc ? ((void *)0) : desc->node->v;  // Line 15
;  n = &desc->node->descs[0];                // Line 16
;  return bar(v, n);
; }

; Line 16, Column 13:
;   n = &desc->node->descs[0];
;              ^

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

%struct.desc = type { %struct.node* }
%struct.node = type { i32*, %struct.desc* }

define i32 @test(%struct.desc* readonly %desc) local_unnamed_addr #0 !dbg !4 {
entry:
  %tobool = icmp eq %struct.desc* %desc, null
  br i1 %tobool, label %cond.end, label %cond.false, !dbg !9
; ALL: br i1 %tobool, label %entry.cond.end_crit_edge, label %cond.false, !dbg [[LOC_15_6:![0-9]+]]
; ALL: entry.cond.end_crit_edge:
; GVN: %.pre = load %struct.node*, %struct.node** null, align 8, !dbg [[LOC_16_13:![0-9]+]]
; INSTCOMBINE:store %struct.node* undef, %struct.node** null, align 536870912, !dbg [[LOC_16_13:![0-9]+]]

cond.false:
  %0 = bitcast %struct.desc* %desc to i8***, !dbg !11
  %1 = load i8**, i8*** %0, align 8, !dbg !11
  %2 = load i8*, i8** %1, align 8
  br label %cond.end, !dbg !9

cond.end:
  %3 = phi i8* [ %2, %cond.false ], [ null, %entry ], !dbg !9
  %node2 = getelementptr inbounds %struct.desc, %struct.desc* %desc, i64 0, i32 0
  %4 = load %struct.node*, %struct.node** %node2, align 8, !dbg !10
  %descs = getelementptr inbounds %struct.node, %struct.node* %4, i64 0, i32 1
  %5 = bitcast %struct.desc** %descs to i8**
  %6 = load i8*, i8** %5, align 8
  %call = tail call i32 @bar(i8* %3, i8* %6)
  ret i32 %call
}

declare i32 @bar(i8*, i8*) local_unnamed_addr #1
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 12, type: !5, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{}
!9 = !DILocation(line: 15, column: 6, scope: !4)
!10 = !DILocation(line: 16, column: 13, scope: !4)
!11 = !DILocation(line: 15, column: 34, scope: !4)

;ALL: [[SCOPE:![0-9]+]] = distinct  !DISubprogram(name: "test",{{.*}}
;ALL: [[LOC_15_6]] = !DILocation(line: 15, column: 6, scope: [[SCOPE]])
;ALL: [[LOC_16_13]] = !DILocation(line: 16, column: 13, scope: [[SCOPE]])
