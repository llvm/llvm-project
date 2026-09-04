; Exercise candidate enumeration on a deep spine ending in a wide set of
; sibling outer/leaf pairs. Pinning the supported depth to the candidates'
; actual depth while setting the attempt budget to seven requires all eight
; candidates to be enumerated. The first seven are attempted in breadth-first
; order, and the eighth anchors the budget remark.
;
; RUN: opt < %s -passes=loop-interchange \
; RUN:     -loop-interchange-min-loop-nest-depth=10 \
; RUN:     -loop-interchange-max-loop-nest-depth=10 \
; RUN:     -loop-interchange-max-inner-subnest-candidates=7 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.yaml -disable-output
; RUN: FileCheck %s --input-file=%t.yaml --implicit-check-not=Interchanged
; RUN: opt < %s -passes='loop(loop-interchange),print<loops>' \
; RUN:     -loop-interchange-min-loop-nest-depth=10 \
; RUN:     -loop-interchange-max-loop-nest-depth=10 \
; RUN:     -loop-interchange-max-inner-subnest-candidates=7 \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=LOOPS

; CHECK-NOT:  --- !
; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 10, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 20, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 30, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 40, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 50, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 60, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 70, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            FallbackCandidateBudget
; CHECK-NEXT: DebugLoc:        { File: inner-subnest-enumeration.ll, Line: 80, Column: 1 }
; CHECK-NEXT: Function:        deep_spine_wide_siblings
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Inner-subnest candidate budget exhausted; the loop nest is left unchanged.'
; CHECK-NEXT: ...
; CHECK-NOT:  --- !

; LOOPS-LABEL: Loop info for function 'deep_spine_wide_siblings':
; LOOPS:         Loop at depth 1 containing: %d1.h<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %d2.h<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %d3.h<header>
; LOOPS-NEXT:          Loop at depth 4 containing: %d4.h<header>
; LOOPS-NEXT:            Loop at depth 5 containing: %d5.h<header>
; LOOPS-NEXT:              Loop at depth 6 containing: %d6.h<header>
; LOOPS-NEXT:                Loop at depth 7 containing: %d7.h<header>
; LOOPS-NEXT:                  Loop at depth 8 containing: %d8.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p0.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p0.i.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p1.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p1.i.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p2.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p2.i.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p3.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p3.i.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p4.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p4.i.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p5.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p5.i.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p6.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p6.i.h<header>
; LOOPS-NEXT:                    Loop at depth 9 containing: %p7.o.h<header>
; LOOPS-NEXT:                      Loop at depth 10 containing: %p7.i.h<header>

define void @deep_spine_wide_siblings() !dbg !5 {
entry:
  br label %d1.h

d1.h:
  %d1 = phi i64 [ 0, %entry ], [ %d1.n, %d1.l ]
  br label %d2.h
d2.h:
  %d2 = phi i64 [ 0, %d1.h ], [ %d2.n, %d2.l ]
  br label %d3.h
d3.h:
  %d3 = phi i64 [ 0, %d2.h ], [ %d3.n, %d3.l ]
  br label %d4.h
d4.h:
  %d4 = phi i64 [ 0, %d3.h ], [ %d4.n, %d4.l ]
  br label %d5.h
d5.h:
  %d5 = phi i64 [ 0, %d4.h ], [ %d5.n, %d5.l ]
  br label %d6.h
d6.h:
  %d6 = phi i64 [ 0, %d5.h ], [ %d6.n, %d6.l ]
  br label %d7.h
d7.h:
  %d7 = phi i64 [ 0, %d6.h ], [ %d7.n, %d7.l ]
  br label %d8.h
d8.h:
  %d8 = phi i64 [ 0, %d7.h ], [ %d8.n, %d8.l ]
  br label %p0.o.h

p0.o.h:
  %p0.o = phi i64 [ 0, %d8.h ], [ %p0.o.n, %p0.o.l ]
  br label %p0.i.h, !dbg !6
p0.i.h:
  %p0.i = phi i64 [ 0, %p0.o.h ], [ %p0.i.n, %p0.i.h ]
  %p0.i.n = add i64 %p0.i, 1
  %p0.i.e = icmp eq i64 %p0.i.n, 2
  br i1 %p0.i.e, label %p0.o.l, label %p0.i.h
p0.o.l:
  %p0.o.n = add i64 %p0.o, 1
  %p0.o.e = icmp eq i64 %p0.o.n, 2
  br i1 %p0.o.e, label %p0.exit, label %p0.o.h
p0.exit:
  br label %p1.o.h

p1.o.h:
  %p1.o = phi i64 [ 0, %p0.exit ], [ %p1.o.n, %p1.o.l ]
  br label %p1.i.h, !dbg !7
p1.i.h:
  %p1.i = phi i64 [ 0, %p1.o.h ], [ %p1.i.n, %p1.i.h ]
  %p1.i.n = add i64 %p1.i, 1
  %p1.i.e = icmp eq i64 %p1.i.n, 2
  br i1 %p1.i.e, label %p1.o.l, label %p1.i.h
p1.o.l:
  %p1.o.n = add i64 %p1.o, 1
  %p1.o.e = icmp eq i64 %p1.o.n, 2
  br i1 %p1.o.e, label %p1.exit, label %p1.o.h
p1.exit:
  br label %p2.o.h

p2.o.h:
  %p2.o = phi i64 [ 0, %p1.exit ], [ %p2.o.n, %p2.o.l ]
  br label %p2.i.h, !dbg !8
p2.i.h:
  %p2.i = phi i64 [ 0, %p2.o.h ], [ %p2.i.n, %p2.i.h ]
  %p2.i.n = add i64 %p2.i, 1
  %p2.i.e = icmp eq i64 %p2.i.n, 2
  br i1 %p2.i.e, label %p2.o.l, label %p2.i.h
p2.o.l:
  %p2.o.n = add i64 %p2.o, 1
  %p2.o.e = icmp eq i64 %p2.o.n, 2
  br i1 %p2.o.e, label %p2.exit, label %p2.o.h
p2.exit:
  br label %p3.o.h

p3.o.h:
  %p3.o = phi i64 [ 0, %p2.exit ], [ %p3.o.n, %p3.o.l ]
  br label %p3.i.h, !dbg !9
p3.i.h:
  %p3.i = phi i64 [ 0, %p3.o.h ], [ %p3.i.n, %p3.i.h ]
  %p3.i.n = add i64 %p3.i, 1
  %p3.i.e = icmp eq i64 %p3.i.n, 2
  br i1 %p3.i.e, label %p3.o.l, label %p3.i.h
p3.o.l:
  %p3.o.n = add i64 %p3.o, 1
  %p3.o.e = icmp eq i64 %p3.o.n, 2
  br i1 %p3.o.e, label %p3.exit, label %p3.o.h
p3.exit:
  br label %p4.o.h

p4.o.h:
  %p4.o = phi i64 [ 0, %p3.exit ], [ %p4.o.n, %p4.o.l ]
  br label %p4.i.h, !dbg !10
p4.i.h:
  %p4.i = phi i64 [ 0, %p4.o.h ], [ %p4.i.n, %p4.i.h ]
  %p4.i.n = add i64 %p4.i, 1
  %p4.i.e = icmp eq i64 %p4.i.n, 2
  br i1 %p4.i.e, label %p4.o.l, label %p4.i.h
p4.o.l:
  %p4.o.n = add i64 %p4.o, 1
  %p4.o.e = icmp eq i64 %p4.o.n, 2
  br i1 %p4.o.e, label %p4.exit, label %p4.o.h
p4.exit:
  br label %p5.o.h

p5.o.h:
  %p5.o = phi i64 [ 0, %p4.exit ], [ %p5.o.n, %p5.o.l ]
  br label %p5.i.h, !dbg !11
p5.i.h:
  %p5.i = phi i64 [ 0, %p5.o.h ], [ %p5.i.n, %p5.i.h ]
  %p5.i.n = add i64 %p5.i, 1
  %p5.i.e = icmp eq i64 %p5.i.n, 2
  br i1 %p5.i.e, label %p5.o.l, label %p5.i.h
p5.o.l:
  %p5.o.n = add i64 %p5.o, 1
  %p5.o.e = icmp eq i64 %p5.o.n, 2
  br i1 %p5.o.e, label %p5.exit, label %p5.o.h
p5.exit:
  br label %p6.o.h

p6.o.h:
  %p6.o = phi i64 [ 0, %p5.exit ], [ %p6.o.n, %p6.o.l ]
  br label %p6.i.h, !dbg !12
p6.i.h:
  %p6.i = phi i64 [ 0, %p6.o.h ], [ %p6.i.n, %p6.i.h ]
  %p6.i.n = add i64 %p6.i, 1
  %p6.i.e = icmp eq i64 %p6.i.n, 2
  br i1 %p6.i.e, label %p6.o.l, label %p6.i.h
p6.o.l:
  %p6.o.n = add i64 %p6.o, 1
  %p6.o.e = icmp eq i64 %p6.o.n, 2
  br i1 %p6.o.e, label %p6.exit, label %p6.o.h
p6.exit:
  br label %p7.o.h

p7.o.h:
  %p7.o = phi i64 [ 0, %p6.exit ], [ %p7.o.n, %p7.o.l ]
  br label %p7.i.h, !dbg !13
p7.i.h:
  %p7.i = phi i64 [ 0, %p7.o.h ], [ %p7.i.n, %p7.i.h ]
  %p7.i.n = add i64 %p7.i, 1
  %p7.i.e = icmp eq i64 %p7.i.n, 2
  br i1 %p7.i.e, label %p7.o.l, label %p7.i.h
p7.o.l:
  %p7.o.n = add i64 %p7.o, 1
  %p7.o.e = icmp eq i64 %p7.o.n, 2
  br i1 %p7.o.e, label %p7.exit, label %p7.o.h
p7.exit:
  br label %d8.l

d8.l:
  %d8.n = add i64 %d8, 1
  %d8.e = icmp eq i64 %d8.n, 2
  br i1 %d8.e, label %d7.l, label %d8.h
d7.l:
  %d7.n = add i64 %d7, 1
  %d7.e = icmp eq i64 %d7.n, 2
  br i1 %d7.e, label %d6.l, label %d7.h
d6.l:
  %d6.n = add i64 %d6, 1
  %d6.e = icmp eq i64 %d6.n, 2
  br i1 %d6.e, label %d5.l, label %d6.h
d5.l:
  %d5.n = add i64 %d5, 1
  %d5.e = icmp eq i64 %d5.n, 2
  br i1 %d5.e, label %d4.l, label %d5.h
d4.l:
  %d4.n = add i64 %d4, 1
  %d4.e = icmp eq i64 %d4.n, 2
  br i1 %d4.e, label %d3.l, label %d4.h
d3.l:
  %d3.n = add i64 %d3, 1
  %d3.e = icmp eq i64 %d3.n, 2
  br i1 %d3.e, label %d2.l, label %d3.h
d2.l:
  %d2.n = add i64 %d2, 1
  %d2.e = icmp eq i64 %d2.n, 2
  br i1 %d2.e, label %d1.l, label %d2.h
d1.l:
  %d1.n = add i64 %d1, 1
  %d1.e = icmp eq i64 %d1.n, 2
  br i1 %d1.e, label %exit, label %d1.h

exit:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "inner-subnest-enumeration.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !2)
!5 = distinct !DISubprogram(name: "deep_spine_wide_siblings", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DILocation(line: 10, column: 1, scope: !5)
!7 = !DILocation(line: 20, column: 1, scope: !5)
!8 = !DILocation(line: 30, column: 1, scope: !5)
!9 = !DILocation(line: 40, column: 1, scope: !5)
!10 = !DILocation(line: 50, column: 1, scope: !5)
!11 = !DILocation(line: 60, column: 1, scope: !5)
!12 = !DILocation(line: 70, column: 1, scope: !5)
!13 = !DILocation(line: 80, column: 1, scope: !5)
