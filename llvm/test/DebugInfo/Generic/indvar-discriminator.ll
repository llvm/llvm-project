; RUN: opt -passes=indvars -S < %s | FileCheck %s
;
; When the induction variable is widened by indvars, check that the debug loc
; associated with the loop increment is correctly propagated.
; Also, when the exit condition of a loop is rewritten to be a canonical !=
; comparison, check that the debug loc of the orginal comparison is reused.
;
; Test case obtained from the following source code:
;
; ////
; extern int foo(int);  // line 1
;
; int bar(int X, int *A) {
;   for (int i = 0; i < X; ++i)  // line 4
;     A[i] = foo(i);
;
;   return A[0];
; }
; ///
;
; Check that the debug location for the loop increment still refers to
; line 4, column 26, discriminator 2.
;
; Check that the canonicalized compare instruction for the loop exit
; condition still refers to line 4, column 21, discriminator 1.
;
; CHECK-LABEL: for.body:
; CHECK: [[PHI:%[0-9a-zA-Z.]+]] = phi i64 [
; CHECK-LABEL: for.inc:
; CHECK: add nuw nsw i64 [[PHI]], 1, !dbg ![[INDVARMD:[0-9]+]]
; CHECK: icmp ne i64 %{{.+}}, %{{.+}}, !dbg ![[ICMPMD:[0-9]+]]
; CHECK-DAG: ![[ICMPMD]] = !DILocation(line: 4, column: 21, scope: ![[ICMPSCOPEMD:[0-9]+]]
; CHECK-DAG: ![[ICMPSCOPEMD]] = !DILexicalBlockFile(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, discriminator: 1)
; CHECK-DAG: ![[INDVARMD]] = !DILocation(line: 4, column: 26, scope: ![[INDVARSCOPEMD:[0-9]+]])
; CHECK-DAG: ![[INDVARSCOPEMD]] = !DILexicalBlockFile(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, discriminator: 2)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @bar(i32 %X, ptr %A) !dbg !5 {
entry:
  %cmp5 = icmp sgt i32 %X, 0, !dbg !7
  br i1 %cmp5, label %for.body.lr.ph, label %for.end, !dbg !9

for.body.lr.ph:
  br label %for.body, !dbg !9

for.body:
  %i.06 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %call = call i32 @foo(i32 %i.06), !dbg !10
  %idxprom = sext i32 %i.06 to i64, !dbg !11
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom, !dbg !11
  store i32 %call, ptr %arrayidx, align 4, !dbg !12
  br label %for.inc, !dbg !11

for.inc:
  %inc = add nsw i32 %i.06, 1, !dbg !13
  %cmp = icmp slt i32 %inc, %X, !dbg !7
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !dbg !9, !llvm.loop !15

for.cond.for.end_crit_edge:
  br label %for.end, !dbg !9

for.end:
  %0 = load i32, ptr %A, align 4, !dbg !17
  ret i32 %0, !dbg !18
}

declare i32 @foo(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 4, column: 21, scope: !8)
!8 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 1)
!9 = !DILocation(line: 4, column: 3, scope: !8)
!10 = !DILocation(line: 5, column: 12, scope: !5)
!11 = !DILocation(line: 5, column: 5, scope: !5)
!12 = !DILocation(line: 5, column: 10, scope: !5)
!13 = !DILocation(line: 4, column: 26, scope: !14)
!14 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 2)
!15 = distinct !{!15, !16}
!16 = !DILocation(line: 4, column: 3, scope: !5)
!17 = !DILocation(line: 7, column: 10, scope: !5)
!18 = !DILocation(line: 7, column: 3, scope: !5)
