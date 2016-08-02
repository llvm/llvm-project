; RUN: opt < %s -S -loop-unroll -unroll-runtime -unroll-count=2 | FileCheck %s

; This tests that setting the unroll count works

; CHECK: for.body.preheader:
; CHECK:   br {{.*}} label %for.body.prol, label %for.body.preheader.split, !dbg [[PH_LOC:![0-9]+]]
; CHECK: for.body.prol:
; CHECK:   br label %for.body.preheader.split, !dbg [[BODY_LOC:![0-9]+]]
; CHECK: for.body.preheader.split:
; CHECK:   br {{.*}} label %for.end.loopexit, label %for.body.preheader.split.split, !dbg [[PH_LOC]]
; CHECK: for.body:
; CHECK:   br i1 %exitcond.1, label %for.end.loopexit.unr-lcssa, label %for.body, !dbg [[BODY_LOC]]
; CHECK-NOT: br i1 %exitcond.4, label %for.end.loopexit{{.*}}, label %for.body

; CHECK-DAG: [[PH_LOC]] = !DILocation(line: 101, column: 1, scope: !{{.*}})
; CHECK-DAG: [[BODY_LOC]] = !DILocation(line: 102, column: 1, scope: !{{.*}})

define i32 @test(i32* nocapture %a, i32 %n) nounwind uwtable readonly !dbg !6 {
entry:
  %cmp1 = icmp eq i32 %n, 0, !dbg !7
  br i1 %cmp1, label %for.end, label %for.body, !dbg !7

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv, !dbg !8
  %0 = load i32, i32* %arrayidx, align 4, !dbg !8
  %add = add nsw i32 %0, %sum.02, !dbg !8
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !9
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !9
  %exitcond = icmp eq i32 %lftr.wideiv, %n, !dbg !9
  br i1 %exitcond, label %for.end, label %for.body, !dbg !9

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa, !dbg !10
}

!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"PIC Level", i32 2}

!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = !DIFile(filename: "test.cpp", directory: "/tmp")
!6 = distinct !DISubprogram(name: "test", scope: !5, file: !5, line: 99, type: !4, isLocal: false, isDefinition: true, scopeLine: 100, flags: DIFlagPrototyped, isOptimized: false, variables: !3)
!7 = !DILocation(line: 100, column: 1, scope: !6)
!8 = !DILocation(line: 101, column: 1, scope: !6)
!9 = !DILocation(line: 102, column: 1, scope: !6)
!10 = !DILocation(line: 103, column: 1, scope: !6)
