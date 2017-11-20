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
  ]

dispatch_op.sw.bb6_crit_edge:                     ; preds = %dispatch_op
  br label %sw.bb6

sw.bb:                                            ; preds = %indirectgoto, %dispatch_op
  br label %exit

TARGET_1:                                         ; preds = %indirectgoto
  br label %sw.bb6
; USE: br label %sw.bb6, !irr_loop {{.*}}

sw.bb6:                                           ; preds = %TARGET_1, %dispatch_op.sw.bb6_crit_edge
  br i1 %tobool, label %dispatch_op, label %if.then
; USE: br i1 %tobool, label %dispatch_op, label %if.then, !prof !{{[0-9]+}},
; USE-SAME: !irr_loop ![[SW_BB6_IRR_LOOP:[0-9]+]]

if.then:                                          ; preds = %sw.bb6
  br label %indirectgoto

TARGET_2:                                         ; preds = %indirectgoto
  br label %sw.bb15
; USE: br label %sw.bb15, !irr_loop {{.*}}

sw.bb15:                                          ; preds = %TARGET_2, %dispatch_op
  br i1 %tobool, label %if.then18, label %exit
; USE: br i1 %tobool, label %if.then18, label %exit, !prof !{{[0-9]+}},
; USE-SAME: !irr_loop ![[SW_BB15_IRR_LOOP:[0-9]+]]

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
  indirectbr i8* %target, [label %unknown_op, label %sw.bb, label %TARGET_1, label %TARGET_2]
; USE: indirectbr i8* %target, [label %unknown_op, label %sw.bb, label %TARGET_1, label %TARGET_2], !prof !{{[0-9]+}},
; USE-SAME: !irr_loop ![[INDIRECTGOTO_IRR_LOOP:[0-9]+]]
}

; USE: ![[FOR_COND2_IRR_LOOP]] = !{!"loop_header_weight", i64 1050}
; USE: ![[ENTRY8_IRR_LOOP]] = !{!"loop_header_weight", i64 373}
; USE: ![[IF_END9_IRR_LOOP]] = !{!"loop_header_weight", i64 1000}
; USE: ![[SW_BB6_IRR_LOOP]] = !{!"loop_header_weight", i64 501}
; USE: ![[SW_BB15_IRR_LOOP]] = !{!"loop_header_weight", i64 100}
; USE: ![[INDIRECTGOTO_IRR_LOOP]] = !{!"loop_header_weight", i64 400}
