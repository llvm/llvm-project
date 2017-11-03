; RUN: llvm-profdata merge %S/Inputs/irreducible.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

; GEN: $__llvm_profile_raw_version = comdat any

; Function Attrs: noinline norecurse nounwind readnone uwtable
define i32 @_Z11irreducibleii(i32 %iter_outer, i32 %iter_inner) local_unnamed_addr #0 {
entry:
  %cmp24 = icmp sgt i32 %iter_outer, 0
  br i1 %cmp24, label %for.body, label %entry.for.cond.cleanup_crit_edge

entry.for.cond.cleanup_crit_edge:                 ; preds = %entry
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %entry.for.cond.cleanup_crit_edge, %for.end
  %sum.0.lcssa = phi i32 [ 0, %entry.for.cond.cleanup_crit_edge ], [ %sum.1, %for.end ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.end
  %k.026 = phi i32 [ %inc12, %for.end ], [ 0, %entry ]
  %sum.025 = phi i32 [ %sum.1, %for.end ], [ 0, %entry ]
  %rem23 = and i32 %k.026, 1
  %cmp1 = icmp eq i32 %rem23, 0
  br i1 %cmp1, label %entry8, label %for.cond2

for.cond2:                                        ; preds = %for.body, %if.end9
  %sum.1 = phi i32 [ %add10, %if.end9 ], [ %sum.025, %for.body ]
  %i.0 = phi i32 [ %inc, %if.end9 ], [ 0, %for.body ]
  %cmp3 = icmp slt i32 %i.0, %iter_inner
  br i1 %cmp3, label %for.body4, label %for.end
; USE: br i1 %cmp3, label %for.body4, label %for.end, !prof !{{[0-9]+}},
; USE-SAME: !irr_loop ![[FOR_COND2_IRR_LOOP:[0-9]+]]

for.body4:                                        ; preds = %for.cond2
  %rem5 = srem i32 %k.026, 3
  %cmp6 = icmp eq i32 %rem5, 0
  br i1 %cmp6, label %entry8, label %if.end9

entry8:                                           ; preds = %for.body4, %for.body
  %sum.2 = phi i32 [ %sum.025, %for.body ], [ %sum.1, %for.body4 ]
  %i.1 = phi i32 [ 0, %for.body ], [ %i.0, %for.body4 ]
  %add = add nsw i32 %sum.2, 4
  br label %if.end9
; USE: br label %if.end9,
; USE-SAME: !irr_loop ![[ENTRY8_IRR_LOOP:[0-9]+]]

if.end9:                                          ; preds = %entry8, %for.body4
  %sum.3 = phi i32 [ %add, %entry8 ], [ %sum.1, %for.body4 ]
  %i.2 = phi i32 [ %i.1, %entry8 ], [ %i.0, %for.body4 ]
  %add10 = add nsw i32 %sum.3, 1
  %inc = add nsw i32 %i.2, 1
  br label %for.cond2
; USE: br label %for.cond2,
; USE-SAME: !irr_loop ![[IF_END9_IRR_LOOP:[0-9]+]]

for.end:                                          ; preds = %for.cond2
  %inc12 = add nuw nsw i32 %k.026, 1
  %exitcond = icmp eq i32 %inc12, %iter_outer
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}



@targets = local_unnamed_addr global [256 x i8*] zeroinitializer, align 16
@tracing = local_unnamed_addr global i32 0, align 4

; Function Attrs: noinline norecurse nounwind uwtable
define i32 @_Z11irreduciblePh(i8* nocapture readonly %p) {
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
  ]

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

sw.bb6:                                           ; preds = %dispatch_op.sw.bb6_crit_edge, %TARGET_1
  %p.addr.2 = phi i8* [ %incdec.ptr4, %TARGET_1 ], [ %p.addr.1, %dispatch_op.sw.bb6_crit_edge ]
  %oparg.2 = phi i32 [ %conv5, %TARGET_1 ], [ %oparg.0, %dispatch_op.sw.bb6_crit_edge ]
  %sum.3 = phi i32 [ %sum.7, %TARGET_1 ], [ %sum.1, %dispatch_op.sw.bb6_crit_edge ]
  %mul = mul nsw i32 %oparg.2, 7
  %add7 = add nsw i32 %sum.3, %mul
  %rem46 = and i32 %add7, 1
  %cmp8 = icmp eq i32 %rem46, 0
  br i1 %cmp8, label %dispatch_op, label %if.then
; USE: br i1 %cmp8, label %dispatch_op, label %if.then, !prof !{{[0-9]+}},
; USE-SAME: !irr_loop ![[SW_BB6_IRR_LOOP:[0-9]+]]

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
  br i1 %tobool, label %if.then18, label %exit
; USE: br i1 %tobool, label %if.then18, label %exit, !prof !{{[0-9]+}},
; USE-SAME: !irr_loop ![[SW_BB15_IRR_LOOP:[0-9]+]]

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
  indirectbr i8* %6, [label %unknown_op, label %sw.bb, label %TARGET_1, label %TARGET_2]
; USE: indirectbr i8* %6, [label %unknown_op, label %sw.bb, label %TARGET_1, label %TARGET_2], !prof !{{[0-9]+}},
; USE-SAME: !irr_loop ![[INDIRECTGOTO_IRR_LOOP:[0-9]+]]
}

; USE: ![[FOR_COND2_IRR_LOOP]] = !{!"loop_header_weight", i64 1050}
; USE: ![[ENTRY8_IRR_LOOP]] = !{!"loop_header_weight", i64 373}
; USE: ![[IF_END9_IRR_LOOP]] = !{!"loop_header_weight", i64 1000}
; USE: ![[SW_BB6_IRR_LOOP]] = !{!"loop_header_weight", i64 501}
; USE: ![[SW_BB15_IRR_LOOP]] = !{!"loop_header_weight", i64 100}
; USE: ![[INDIRECTGOTO_IRR_LOOP]] = !{!"loop_header_weight", i64 400}
