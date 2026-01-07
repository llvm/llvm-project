
; RUN: opt %s -S --passes=loop-unroll -unroll-runtime=true -unroll-count=4 -unroll-remainder -o - \
; RUN: | FileCheck %s

;; Check atoms are remapped for runtime unrolling.

; CHECK: for.body.epil:
; CHECK-NEXT: store i64 %indvars.iv.epil.init, ptr %p, align 4, !dbg [[G2R1:!.*]]

; CHECK: for.body.epil.1:
; CHECK-NEXT: store i64 %indvars.iv.next.epil, ptr %p, align 4, !dbg [[G3R1:!.*]]

; CHECK: for.body.epil.2:
; CHECK-NEXT: store i64 %indvars.iv.next.epil.1, ptr %p, align 4, !dbg [[G4R1:!.*]]

; CHECK: for.body:
; CHECK-NEXT: %indvars.iv = phi i64 [ 0, %for.body.lr.ph.new ], [ %indvars.iv.next.3, %for.body ]
; CHECK-NEXT: %niter = phi i64 [ 0, %for.body.lr.ph.new ], [ %niter.next.3, %for.body ]
; CHECK-NEXT: store i64 %indvars.iv, ptr %p, align 4, !dbg [[G1R1:!.*]]
; CHECK-NEXT: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT: store i64 %indvars.iv.next, ptr %p, align 4, !dbg [[G5R1:!.*]]
; CHECK-NEXT: %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2
; CHECK-NEXT: store i64 %indvars.iv.next.1, ptr %p, align 4, !dbg [[G6R1:!.*]]
; CHECK-NEXT: %indvars.iv.next.2 = add nuw nsw i64 %indvars.iv, 3
; CHECK-NEXT: store i64 %indvars.iv.next.2, ptr %p, align 4, !dbg [[G7R1:!.*]]
; CHECK-NEXT: %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 4

; CHECK: distinct !DISubprogram(name: "unroll", {{.*}}keyInstructions: true)
; CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
; CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
; CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
; CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
; CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
; CHECK: [[G6R1]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)
; CHECK: [[G7R1]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 1)

define i32 @unroll(ptr %p, i32 %N) local_unnamed_addr !dbg !5 {
entry:
  %cmp9 = icmp eq i32 %N, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %r = phi i32 [ 0, %entry ], [ 1, %for.body ]
  ret i32 %r

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  store i64 %indvars.iv, ptr %p, !dbg !8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 17}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "unroll", linkageName: "unroll", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5, atomGroup: 1, atomRank: 1)
