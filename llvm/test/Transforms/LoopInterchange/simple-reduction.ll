; NOTE: Support simple reduction in the inner loop by undoing the simple reduction.
; RUN: opt < %s -passes="loop(loop-interchange),dce"  -undo-simple-reduction -loop-interchange-profitabilities=ignore -S | FileCheck %s

; for (int i = 0; i < n; i++) {
;   s[i] = 0;
;   for (int j = 0; j < n; j++)
;     s[i] = s[i] + a[j][i] * b[j][i];
; }

define void @func(ptr noalias noundef readonly captures(none) %a, ptr noalias noundef readonly captures(none) %b, ptr noalias noundef writeonly captures(none) %s, i64  noundef %n) {
; CHECK-LABEL: define void @func(ptr noalias noundef readonly captures(none) %a, ptr noalias noundef readonly captures(none) %b, ptr noalias noundef writeonly captures(none) %s, i64  noundef %n) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i64 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[INNERLOOP_PREHEADER:%.*]], label [[EXIT:%.*]]
; CHECK:       outerloop_header.preheader:
; CHECK-NEXT:    br label [[OUTERLOOP_HEADER:%.*]]
; CHECK:       outerloop_header:
; CHECK-NEXT:    [[INDEX_I:%.*]] = phi i64 [ [[I_NEXT:%.*]], [[OUTERLOOP_LATCH:%.*]] ], [ 0, [[OUTERLOOPHEADER_PREHEADER:%.*]] ]
; CHECK-NEXT:    [[ADDR_S:%.*]] = getelementptr inbounds nuw double, ptr %s, i64 [[INDEX_I]]
; CHECK-NEXT:    [[ADDR_A:%.*]] = getelementptr inbounds nuw [100 x double], ptr %a, i64 0, i64 [[INDEX_I]]
; CHECK-NEXT:    [[ADDR_B:%.*]] = getelementptr inbounds nuw [100 x double], ptr %b, i64 0, i64 [[INDEX_I]]
; CHECK-NEXT:    br label [[INNERLOOP_SPLIT1:%.*]]
; CHECK:       innerloop.preheader:
; CHECK-NEXT:    br label [[INNERLOOP:%.*]]
; CHECK:       innerloop:
; CHECK-NEXT:    [[INDEX_J:%.*]] = phi i64 [ [[J_NEXT:%.*]], [[INNERLOOP_SPLIT:%.*]] ], [ 0, [[INNERLOOP_PREHEADER:%.*]] ]
; CHECK-NEXT:    br label [[OUTERLOOPHEADER_PREHEADER:%.*]]
; CHECK:       innerloop.split1:
; CHECK-NEXT:    [[S:%.*]] = load double, ptr [[ADDR_S]], align 8
; CHECK-NEXT:    [[FIRSTITER:%.*]] = icmp ne i64 [[INDEX_J]], 0
; CHECK-NEXT:    [[NEW_VAR:%.*]] = select i1 [[FIRSTITER]], double [[S]], double 0.000000e+00
; CHECK-NEXT:    [[ADDR_A_J_I:%.*]] = getelementptr inbounds nuw [100 x double], ptr [[ADDR_A]], i64 [[INDEX_J]]
; CHECK-NEXT:    [[A_J_I:%.*]] = load double, ptr [[ADDR_A_J_I]], align 8
; CHECK-NEXT:    [[ADDR_B_J_I:%.*]] = getelementptr inbounds nuw [100 x double], ptr [[ADDR_B]], i64 [[INDEX_J]]
; CHECK-NEXT:    [[B_J_I:%.*]] = load double, ptr [[ADDR_B_J_I]], align 8
; CHECK-NEXT:    [[MUL:%.*]] = fmul fast double [[B_J_I]], [[A_J_I]]
; CHECK-NEXT:    [[ADD:%.*]] = fadd fast double [[MUL]], [[NEW_VAR]]
; CHECK-NEXT:    store double [[ADD]], ptr [[ADDR_S]], align 8
; CHECK-NEXT:    br label [[OUTERLOOP_LATCH:%.*]]
; CHECK:       innerloop.split:
; CHECK-NEXT:    [[J_NEXT:%.*]] = add nuw nsw i64 [[INDEX_J]], 1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i64 [[J_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT_LOOPEXIT:%.*]], label [[INNERLOOP]]
; CHECK:       outerloop_latch:
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i64 [[INDEX_I]], 1
; CHECK-NEXT:    [[CMP2:%.*]] = icmp eq i64 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP2]], label [[INNERLOOP_SPLIT:%.*]], label [[OUTERLOOP_HEADER]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %outerloop_header, label %exit

outerloop_header:                                      
  %index_i = phi i64 [ 0, %entry ], [ %index_i.next, %outerloop_latch ]
  %addr_s = getelementptr inbounds nuw double, ptr %s, i64 %index_i
  %invariant.gep.us = getelementptr inbounds nuw [100 x double], ptr %a, i64 0, i64 %index_i
  %invariant.gep32.us = getelementptr inbounds nuw [100 x double], ptr %b, i64 0, i64 %index_i
  br label %innerloop

innerloop:                                     
  %index_j = phi i64 [ 0, %outerloop_header ], [ %index_j.next, %innerloop ]
  %reduction = phi double [ 0.000000e+00, %outerloop_header ], [ %add, %innerloop ]
  %addr_a_j_i = getelementptr inbounds nuw [100 x double], ptr %invariant.gep.us, i64 %index_j
  %0 = load double, ptr %addr_a_j_i, align 8
  %addr_b_j_i = getelementptr inbounds nuw [100 x double], ptr %invariant.gep32.us, i64 %index_j
  %1 = load double, ptr %addr_b_j_i, align 8
  %mul = fmul fast double %1, %0
  %add = fadd fast double %mul, %reduction
  %index_j.next = add nuw nsw i64 %index_j, 1
  %cond1 = icmp eq i64 %index_j.next, %n
  br i1 %cond1, label %outerloop_latch, label %innerloop

outerloop_latch:         
  %lcssa = phi double [ %add, %innerloop ]
  store double %lcssa, ptr %addr_s, align 8
  %index_i.next = add nuw nsw i64 %index_i, 1
  %cond2 = icmp eq i64 %index_i.next, %n
  br i1 %cond2, label %exit, label %outerloop_header

exit:                                 
  ret void
}
