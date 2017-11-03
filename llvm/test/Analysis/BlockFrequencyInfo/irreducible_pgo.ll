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
  store <2 x i8*> <i8* blockaddress(@_Z11irreduciblePh, %sw.bb), i8* blockaddress(@_Z11irreduciblePh, %TARGET_1)>, <2 x i8*>* bitcast ([256 x i8*]* @targets to <2 x i8*>*), align 16
  store i8* blockaddress(@_Z11irreduciblePh, %TARGET_2), i8** getelementptr inbounds ([256 x i8*], [256 x i8*]* @targets, i64 0, i64 2), align 16
  %0 = load i32, i32* @tracing, align 4
  %tobool = icmp eq i32 %0, 0
  br label %for.cond1

for.cond1:                                        ; preds = %sw.default, %entry
  %p.addr.0 = phi i8* [ %p, %entry ], [ %p.addr.4, %sw.default ]
  %sum.0 = phi i32 [ 0, %entry ], [ %add25, %sw.default ]
  %incdec.ptr = getelementptr inbounds i8, i8* %p.addr.0, i64 1
  %1 = load i8, i8* %p.addr.0, align 1
  %incdec.ptr2 = getelementptr inbounds i8, i8* %p.addr.0, i64 2
  %2 = load i8, i8* %incdec.ptr, align 1
  %conv3 = zext i8 %2 to i32
  br label %dispatch_op

dispatch_op:                                      ; preds = %sw.bb6, %for.cond1
  %p.addr.1 = phi i8* [ %incdec.ptr2, %for.cond1 ], [ %p.addr.2, %sw.bb6 ]
  %op.0 = phi i8 [ %1, %for.cond1 ], [ 1, %sw.bb6 ]
  %oparg.0 = phi i32 [ %conv3, %for.cond1 ], [ %oparg.2, %sw.bb6 ]
  %sum.1 = phi i32 [ %sum.0, %for.cond1 ], [ %add7, %sw.bb6 ]
  switch i8 %op.0, label %sw.default [
    i8 0, label %sw.bb
    i8 1, label %dispatch_op.sw.bb6_crit_edge
    i8 2, label %sw.bb15
  ], !prof !36

dispatch_op.sw.bb6_crit_edge:                     ; preds = %dispatch_op
  br label %sw.bb6

sw.bb:                                            ; preds = %indirectgoto, %dispatch_op
  %oparg.1 = phi i32 [ %oparg.0, %dispatch_op ], [ 0, %indirectgoto ]
  %sum.2 = phi i32 [ %sum.1, %dispatch_op ], [ %sum.7, %indirectgoto ]
  %add.neg = sub i32 -5, %oparg.1
  %sub = add i32 %add.neg, %sum.2
  br label %exit

TARGET_1:                                         ; preds = %indirectgoto
  %incdec.ptr4 = getelementptr inbounds i8, i8* %add.ptr.pn, i64 2
  %3 = load i8, i8* %p.addr.5, align 1
  %conv5 = zext i8 %3 to i32
  br label %sw.bb6

sw.bb6:                                           ; preds = %TARGET_1, %dispatch_op.sw.bb6_crit_edge
  %p.addr.2 = phi i8* [ %incdec.ptr4, %TARGET_1 ], [ %p.addr.1, %dispatch_op.sw.bb6_crit_edge ]
  %oparg.2 = phi i32 [ %conv5, %TARGET_1 ], [ %oparg.0, %dispatch_op.sw.bb6_crit_edge ]
  %sum.3 = phi i32 [ %sum.7, %TARGET_1 ], [ %sum.1, %dispatch_op.sw.bb6_crit_edge ]
  %mul = mul nsw i32 %oparg.2, 7
  %add7 = add nsw i32 %sum.3, %mul
  %rem46 = and i32 %add7, 1
  %cmp8 = icmp eq i32 %rem46, 0
  br i1 %cmp8, label %dispatch_op, label %if.then, !prof !37, !irr_loop !38

if.then:                                          ; preds = %sw.bb6
  %mul9 = mul nsw i32 %add7, 9
  br label %indirectgoto

TARGET_2:                                         ; preds = %indirectgoto
  %incdec.ptr13 = getelementptr inbounds i8, i8* %add.ptr.pn, i64 2
  %4 = load i8, i8* %p.addr.5, align 1
  %conv14 = zext i8 %4 to i32
  br label %sw.bb15

sw.bb15:                                          ; preds = %TARGET_2, %dispatch_op
  %p.addr.3 = phi i8* [ %p.addr.1, %dispatch_op ], [ %incdec.ptr13, %TARGET_2 ]
  %oparg.3 = phi i32 [ %oparg.0, %dispatch_op ], [ %conv14, %TARGET_2 ]
  %sum.4 = phi i32 [ %sum.1, %dispatch_op ], [ %sum.7, %TARGET_2 ]
  %add16 = add nsw i32 %oparg.3, 3
  %add17 = add nsw i32 %add16, %sum.4
  br i1 %tobool, label %if.then18, label %exit, !prof !39, !irr_loop !40

if.then18:                                        ; preds = %sw.bb15
  %idx.ext = sext i32 %oparg.3 to i64
  %add.ptr = getelementptr inbounds i8, i8* %p.addr.3, i64 %idx.ext
  %mul19 = mul nsw i32 %add17, 17
  br label %indirectgoto

unknown_op:                                       ; preds = %indirectgoto
  %sub24 = add nsw i32 %sum.7, -4
  br label %sw.default

sw.default:                                       ; preds = %unknown_op, %dispatch_op
  %p.addr.4 = phi i8* [ %p.addr.5, %unknown_op ], [ %p.addr.1, %dispatch_op ]
  %sum.5 = phi i32 [ %sub24, %unknown_op ], [ %sum.1, %dispatch_op ]
  %add25 = add nsw i32 %sum.5, 11
  br label %for.cond1

exit:                                             ; preds = %sw.bb15, %sw.bb
  %sum.6 = phi i32 [ %sub, %sw.bb ], [ %add17, %sw.bb15 ]
  ret i32 %sum.6

indirectgoto:                                     ; preds = %if.then18, %if.then
  %add.ptr.pn = phi i8* [ %add.ptr, %if.then18 ], [ %p.addr.2, %if.then ]
  %sum.7 = phi i32 [ %mul19, %if.then18 ], [ %mul9, %if.then ]
  %p.addr.5 = getelementptr inbounds i8, i8* %add.ptr.pn, i64 1
  %5 = load i8, i8* %add.ptr.pn, align 1
  %idxprom21 = zext i8 %5 to i64
  %arrayidx22 = getelementptr inbounds [256 x i8*], [256 x i8*]* @targets, i64 0, i64 %idxprom21
  %6 = load i8*, i8** %arrayidx22, align 8
  indirectbr i8* %6, [label %unknown_op, label %sw.bb, label %TARGET_1, label %TARGET_2], !prof !41, !irr_loop !42
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
