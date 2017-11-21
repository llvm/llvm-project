; RUN: opt < %s -analyze -block-freq | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; Function Attrs: noinline norecurse nounwind readnone uwtable
define i32 @_Z11irreducibleii(i32 %iter_outer, i32 %iter_inner) local_unnamed_addr !prof !27 {
entry:
  %cmp24 = icmp sgt i32 %iter_outer, 0
  br i1 %cmp24, label %for.body, label %entry.for.cond.cleanup_crit_edge, !prof !28

entry.for.cond.cleanup_crit_edge:                 ; preds = %entry
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.end, %entry.for.cond.cleanup_crit_edge
  %sum.0.lcssa = phi i32 [ 0, %entry.for.cond.cleanup_crit_edge ], [ %sum.1, %for.end ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %for.end, %entry
  %k.026 = phi i32 [ %inc12, %for.end ], [ 0, %entry ]
  %sum.025 = phi i32 [ %sum.1, %for.end ], [ 0, %entry ]
  %rem23 = and i32 %k.026, 1
  %cmp1 = icmp eq i32 %rem23, 0
  br i1 %cmp1, label %entry8, label %for.cond2, !prof !29

for.cond2:                                        ; preds = %if.end9, %for.body
  %sum.1 = phi i32 [ %add10, %if.end9 ], [ %sum.025, %for.body ]
  %i.0 = phi i32 [ %inc, %if.end9 ], [ 0, %for.body ]
  %cmp3 = icmp slt i32 %i.0, %iter_inner
  br i1 %cmp3, label %for.body4, label %for.end, !prof !30, !irr_loop !31

for.body4:                                        ; preds = %for.cond2
  %rem5 = srem i32 %k.026, 3
  %cmp6 = icmp eq i32 %rem5, 0
  br i1 %cmp6, label %entry8, label %if.end9, !prof !32

entry8:                                           ; preds = %for.body4, %for.body
  %sum.2 = phi i32 [ %sum.025, %for.body ], [ %sum.1, %for.body4 ]
  %i.1 = phi i32 [ 0, %for.body ], [ %i.0, %for.body4 ]
  %add = add nsw i32 %sum.2, 4
  br label %if.end9, !irr_loop !33

if.end9:                                          ; preds = %entry8, %for.body4
  %sum.3 = phi i32 [ %add, %entry8 ], [ %sum.1, %for.body4 ]
  %i.2 = phi i32 [ %i.1, %entry8 ], [ %i.0, %for.body4 ]
  %add10 = add nsw i32 %sum.3, 1
  %inc = add nsw i32 %i.2, 1
  br label %for.cond2, !irr_loop !34

for.end:                                          ; preds = %for.cond2
  %inc12 = add nuw nsw i32 %k.026, 1
  %exitcond = icmp eq i32 %inc12, %iter_outer
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !prof !35
}

!27 = !{!"function_entry_count", i64 1}
!28 = !{!"branch_weights", i32 1, i32 0}
!29 = !{!"branch_weights", i32 50, i32 50}
!30 = !{!"branch_weights", i32 950, i32 100}
!31 = !{!"loop_header_weight", i64 1050}
!32 = !{!"branch_weights", i32 323, i32 627}
!33 = !{!"loop_header_weight", i64 373}
!34 = !{!"loop_header_weight", i64 1000}
!35 = !{!"branch_weights", i32 1, i32 99}

; CHECK-LABEL: Printing analysis {{.*}} for function '_Z11irreducibleii':
; CHECK-NEXT: block-frequency-info: _Z11irreducibleii
; CHECK-NEXT: - entry: {{.*}} count = 1
; CHECK-NEXT: - entry.for.cond.cleanup_crit_edge: {{.*}} count = 0
; CHECK-NEXT: - for.cond.cleanup: {{.*}} count = 1
; CHECK-NEXT: - for.body: {{.*}} count = 100
; CHECK-NEXT: - for.cond2: {{.*}} count = 1050, irr_loop_header_weight = 1050
; CHECK-NEXT: - for.body4: {{.*}} count = 950
; CHECK-NEXT: - entry8: {{.*}} count = 373, irr_loop_header_weight = 373
; CHECK-NEXT: - if.end9: {{.*}} count = 1000, irr_loop_header_weight = 1000
; CHECK-NEXT: - for.end: {{.*}} count = 100

@targets = local_unnamed_addr global [256 x i8*] zeroinitializer, align 16
@tracing = local_unnamed_addr global i32 0, align 4

; Function Attrs: noinline norecurse nounwind uwtable
define i32 @_Z11irreduciblePh(i8* nocapture readonly %p) !prof !27 {
entry:
  %0 = load i32, i32* @tracing, align 4
  %1 = trunc i32 %0 to i8
  %tobool = icmp eq i32 %0, 0
  br label %for.cond1

for.cond1:                                        ; preds = %sw.default, %entry
  br label %dispatch_op

dispatch_op:                                      ; preds = %sw.bb6, %for.cond1
  switch i8 %1, label %sw.default [
    i8 0, label %sw.bb
    i8 1, label %dispatch_op.sw.bb6_crit_edge
    i8 2, label %sw.bb15
  ], !prof !36

dispatch_op.sw.bb6_crit_edge:                     ; preds = %dispatch_op
  br label %sw.bb6

sw.bb:                                            ; preds = %indirectgoto, %dispatch_op
  br label %exit

TARGET_1:                                         ; preds = %indirectgoto
  br label %sw.bb6

sw.bb6:                                           ; preds = %TARGET_1, %dispatch_op.sw.bb6_crit_edge
  br i1 %tobool, label %dispatch_op, label %if.then, !prof !37, !irr_loop !38

if.then:                                          ; preds = %sw.bb6
  br label %indirectgoto

TARGET_2:                                         ; preds = %indirectgoto
  br label %sw.bb15

sw.bb15:                                          ; preds = %TARGET_2, %dispatch_op
  br i1 %tobool, label %if.then18, label %exit, !prof !39, !irr_loop !40

if.then18:                                        ; preds = %sw.bb15
  br label %indirectgoto

unknown_op:                                       ; preds = %indirectgoto
  br label %sw.default

sw.default:                                       ; preds = %unknown_op, %dispatch_op
  br label %for.cond1

exit:                                             ; preds = %sw.bb15, %sw.bb
  ret i32 0

indirectgoto:                                     ; preds = %if.then18, %if.then
  %idxprom21 = zext i32 %0 to i64
  %arrayidx22 = getelementptr inbounds [256 x i8*], [256 x i8*]* @targets, i64 0, i64 %idxprom21
  %target = load i8*, i8** %arrayidx22, align 8
  indirectbr i8* %target, [label %unknown_op, label %sw.bb, label %TARGET_1, label %TARGET_2], !prof !41, !irr_loop !42
}

!36 = !{!"branch_weights", i32 0, i32 0, i32 201, i32 1}
!37 = !{!"branch_weights", i32 201, i32 300}
!38 = !{!"loop_header_weight", i64 501}
!39 = !{!"branch_weights", i32 100, i32 0}
!40 = !{!"loop_header_weight", i64 100}
!41 = !{!"branch_weights", i32 0, i32 1, i32 300, i32 99}
!42 = !{!"loop_header_weight", i64 400}

; CHECK-LABEL: Printing analysis {{.*}} for function '_Z11irreduciblePh':
; CHECK-NEXT: block-frequency-info: _Z11irreduciblePh
; CHECK-NEXT: - entry: {{.*}} count = 1
; CHECK-NEXT: - for.cond1: {{.*}} count = 1
; CHECK-NEXT: - dispatch_op: {{.*}} count = 201
; CHECK-NEXT: - dispatch_op.sw.bb6_crit_edge: {{.*}} count = 200
; CHECK-NEXT: - sw.bb: {{.*}} count = 0
; CHECK-NEXT: - TARGET_1: {{.*}} count = 299
; CHECK-NEXT: - sw.bb6: {{.*}} count = 500, irr_loop_header_weight = 501
; CHECK-NEXT: - if.then: {{.*}} count = 299
; CHECK-NEXT: - TARGET_2: {{.*}} count = 98
; CHECK-NEXT: - sw.bb15: {{.*}} count = 99, irr_loop_header_weight = 100
; CHECK-NEXT: - if.then18: {{.*}} count = 99
; CHECK-NEXT: - unknown_op: {{.*}} count = 0
; CHECK-NEXT: - sw.default: {{.*}} count = 0
; CHECK-NEXT: - exit: {{.*}} count = 1
; CHECK-NEXT: - indirectgoto: {{.*}} count = 399, irr_loop_header_weight = 400

; Missing some irr loop annotations.
; Function Attrs: noinline norecurse nounwind uwtable
define i32 @_Z11irreduciblePh2(i8* nocapture readonly %p) !prof !27 {
entry:
  %0 = load i32, i32* @tracing, align 4
  %1 = trunc i32 %0 to i8
  %tobool = icmp eq i32 %0, 0
  br label %for.cond1

for.cond1:                                        ; preds = %sw.default, %entry
  br label %dispatch_op

dispatch_op:                                      ; preds = %sw.bb6, %for.cond1
switch i8 %1, label %sw.default [
    i8 0, label %sw.bb
    i8 1, label %dispatch_op.sw.bb6_crit_edge
    i8 2, label %sw.bb15
  ], !prof !36

dispatch_op.sw.bb6_crit_edge:                     ; preds = %dispatch_op
  br label %sw.bb6

sw.bb:                                            ; preds = %indirectgoto, %dispatch_op
  br label %exit

TARGET_1:                                         ; preds = %indirectgoto
  br label %sw.bb6

sw.bb6:                                           ; preds = %TARGET_1, %dispatch_op.sw.bb6_crit_edge
  br i1 %tobool, label %dispatch_op, label %if.then, !prof !37  ; Missing !irr_loop !38

if.then:                                          ; preds = %sw.bb6
  br label %indirectgoto

TARGET_2:                                         ; preds = %indirectgoto
  br label %sw.bb15

sw.bb15:                                          ; preds = %TARGET_2, %dispatch_op
  br i1 %tobool, label %if.then18, label %exit, !prof !39, !irr_loop !40

if.then18:                                        ; preds = %sw.bb15
  br label %indirectgoto

unknown_op:                                       ; preds = %indirectgoto
  br label %sw.default

sw.default:                                       ; preds = %unknown_op, %dispatch_op
  br label %for.cond1

exit:                                             ; preds = %sw.bb15, %sw.bb
  ret i32 0

indirectgoto:                                     ; preds = %if.then18, %if.then
  %idxprom21 = zext i32 %0 to i64
  %arrayidx22 = getelementptr inbounds [256 x i8*], [256 x i8*]* @targets, i64 0, i64 %idxprom21
  %target = load i8*, i8** %arrayidx22, align 8
  indirectbr i8* %target, [label %unknown_op, label %sw.bb, label %TARGET_1, label %TARGET_2], !prof !41, !irr_loop !42
}

; CHECK-LABEL: Printing analysis {{.*}} for function '_Z11irreduciblePh2':
; CHECK: block-frequency-info: _Z11irreduciblePh2
; CHECK: - sw.bb6: {{.*}} count = 100
; CHECK: - sw.bb15: {{.*}} count = 100, irr_loop_header_weight = 100
; CHECK: - indirectgoto: {{.*}} count = 400, irr_loop_header_weight = 400
