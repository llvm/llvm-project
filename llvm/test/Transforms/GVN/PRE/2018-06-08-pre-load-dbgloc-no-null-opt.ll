; This test checks if debug loc is propagated to load/store created by GVN/Instcombine.
; RUN: opt < %s -passes=gvn -S | FileCheck %s --check-prefixes=ALL
; RUN: opt < %s -passes=gvn,instcombine -S | FileCheck %s --check-prefixes=ALL

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

%struct.desc = type { ptr }
%struct.node = type { ptr, ptr }

define i32 @test_no_null_opt(ptr readonly %desc) local_unnamed_addr #0 !dbg !4 {
entry:
  %tobool = icmp eq ptr %desc, null
  br i1 %tobool, label %cond.end, label %cond.false, !dbg !9
; ALL: %.pre = load ptr, ptr %desc, align 8, !dbg [[LOC_16_13:![0-9]+]]
; ALL: br i1 %tobool, label %cond.end, label %cond.false, !dbg [[LOC_15_6:![0-9]+]]
; ALL: cond.false:

cond.false:
  %0 = load ptr, ptr %desc, align 8, !dbg !11
  %1 = load ptr, ptr %0, align 8
  br label %cond.end, !dbg !9

cond.end:
; ALL: phi ptr [ %0, %cond.false ], [ null, %entry ]

  %2 = phi ptr [ %1, %cond.false ], [ null, %entry ], !dbg !9
  %3 = load ptr, ptr %desc, align 8, !dbg !10
  %descs = getelementptr inbounds %struct.node, ptr %3, i64 0, i32 1
  %4 = load ptr, ptr %descs, align 8
  %call = tail call i32 @bar(ptr %2, ptr %4)
  ret i32 %call
}
attributes #0 = { null_pointer_is_valid }

declare i32 @bar(ptr, ptr) local_unnamed_addr #1
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test_no_null_opt", scope: !1, file: !1, line: 12, type: !5, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{}
!9 = !DILocation(line: 15, column: 6, scope: !4)
!10 = !DILocation(line: 16, column: 13, scope: !4)
!11 = !DILocation(line: 15, column: 34, scope: !4)

;ALL: [[SCOPE:![0-9]+]] = distinct  !DISubprogram(name: "test_no_null_opt",{{.*}}
;ALL: [[LOC_16_13]] = !DILocation(line: 16, column: 13, scope: [[SCOPE]])
;ALL: [[LOC_15_6]] = !DILocation(line: 15, column: 6, scope: [[SCOPE]])
