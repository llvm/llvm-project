; RUN: opt < %s -S -loop-unroll -unroll-runtime -unroll-count=2 -unroll-runtime-epilog=true | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -loop-unroll -unroll-runtime -unroll-count=2 -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG

; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop-unroll' -unroll-runtime -unroll-count=2 -unroll-runtime-epilog=true | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop-unroll' -unroll-runtime -unroll-count=2 -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG

; This tests that setting the unroll count works


; EPILOG: for.body.preheader:
; EPILOG:   br i1 %1, label %for.end.loopexit.unr-lcssa, label %for.body.preheader.new, !dbg [[PH_LOC:![0-9]+]]
; EPILOG: for.body:
; EPILOG:   br i1 %niter.ncmp.1, label %for.end.loopexit.unr-lcssa.loopexit, label %for.body, !dbg [[PH_LOC]]
; EPILOG-NOT: br i1 %niter.ncmp.2, label %for.end.loopexit{{.*}}, label %for.body
; EPILOG: for.body.epil.preheader:
; EPILOG:   br label %for.body.epil, !dbg [[PH_LOC]]
; EPILOG: for.body.epil:
; EPILOG:   br label %for.end.loopexit.epilog-lcssa, !dbg [[PH_LOC]]
; EPILOG: for.end.loopexit:
; EPILOG:   br label %for.end, !dbg [[EXIT_LOC:![0-9]+]]

; EPILOG-DAG: [[PH_LOC]] = !DILocation(line: 102, column: 1, scope: !{{.*}})
; EPILOG-DAG: [[EXIT_LOC]] = !DILocation(line: 103, column: 1, scope: !{{.*}})

; PROLOG: for.body.preheader:
; PROLOG:   br {{.*}} label %for.body.prol.preheader, label %for.body.prol.loopexit, !dbg [[PH_LOC:![0-9]+]]
; PROLOG: for.body.prol:
; PROLOG:   br label %for.body.prol.loopexit, !dbg [[PH_LOC:![0-9]+]]
; PROLOG: for.body.prol.loopexit:
; PROLOG:   br {{.*}} label %for.end.loopexit, label %for.body.preheader.new, !dbg [[PH_LOC]]
; PROLOG: for.body:
; PROLOG:   br i1 %exitcond.1, label %for.end.loopexit.unr-lcssa, label %for.body, !dbg [[PH_LOC]]
; PROLOG-NOT: br i1 %exitcond.4, label %for.end.loopexit{{.*}}, label %for.body

; PROLOG-DAG: [[PH_LOC]] = !DILocation(line: 102, column: 1, scope: !{{.*}})

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
!llvm.dbg.cu = !{!11}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"PIC Level", i32 2}

!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = !DIFile(filename: "test.cpp", directory: "/tmp")
!6 = distinct !DISubprogram(name: "test", scope: !5, file: !5, line: 99, type: !4, isLocal: false, isDefinition: true, scopeLine: 100, flags: DIFlagPrototyped, isOptimized: false, unit: !11, retainedNodes: !3)
!7 = !DILocation(line: 100, column: 1, scope: !6)
!8 = !DILocation(line: 101, column: 1, scope: !6)
!9 = !DILocation(line: 102, column: 1, scope: !6)
!10 = !DILocation(line: 103, column: 1, scope: !6)
!11 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !5,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2) 
