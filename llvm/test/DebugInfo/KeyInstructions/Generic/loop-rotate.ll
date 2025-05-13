; RUN: opt --passes=loop-rotate %s -S -o - | FileCheck %s

;; Rotate:
;;                +------------------> for.end.
;;                |
;;    entry -> for.cond -> for.body
;;                ^           |
;;                +-----------+
;;
;; Into:
;;
;;                               +------> for.end.
;;                               |
;;    entry (+ for.cond`0) -> for.body (+ for.cond) -+
;;                               ^                   |
;;                               +-------------------+
;; Check for.cond's duplicated store and br have their source atoms remapped.

; CHECK: entry:
; CHECK:   store i32 0, ptr @glob, align 16, !dbg [[G3R1:![0-9]+]]
; CHECK:   br label %for.body, !dbg [[G4R1:![0-9]+]]
;
; CHECK: for.body:
; CHECK:    store i32 {{.*}}, ptr @glob, align 16, !dbg [[G1R1:![0-9]+]]
; CHECK:    [[CMP:%.*]] = icmp slt i32 {{.*}}, 100, !dbg [[G2R2:![0-9]+]]
; CHECK:    br i1 [[CMP]], label %for.body, label %for.end, !dbg [[G2R1:![0-9]+]]
;
; CHECK: [[G3R1]] = !DILocation(line: 4{{.*}}, atomGroup: 3, atomRank: 1)
; CHECK: [[G4R1]] = !DILocation(line: 6{{.*}}, atomGroup: 4, atomRank: 1)
; CHECK: [[G1R1]] = !DILocation(line: 4{{.*}}, atomGroup: 1, atomRank: 1)
; CHECK: [[G2R2]] = !DILocation(line: 5{{.*}}, atomGroup: 2, atomRank: 2)
; CHECK: [[G2R1]] = !DILocation(line: 6{{.*}}, atomGroup: 2, atomRank: 1)

@glob = global i32 0

define void @test1() #0 !dbg !5 {
entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  store i32 %i.0, ptr @glob, align 16,         !dbg !DILocation(line: 4, scope: !5, atomGroup: 1, atomRank: 1)
  %cmp = icmp slt i32 %i.0, 100,               !dbg !DILocation(line: 5, scope: !5, atomGroup: 2, atomRank: 2)
  br i1 %cmp, label %for.body, label %for.end, !dbg !DILocation(line: 6, scope: !5, atomGroup: 2, atomRank: 1)

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %i.0, 1
  store i32 0, ptr %array, align 16
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arrayidx.lcssa = phi ptr [ %array, %for.cond ]
  call void @g(ptr %arrayidx.lcssa)
  ret void
}

declare void @g(ptr)

attributes #0 = { nounwind ssp }
attributes #1 = { noduplicate }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 12}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
