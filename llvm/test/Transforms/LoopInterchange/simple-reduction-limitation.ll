; Several cases of undoing simple reductions that have not yet been supported.
; RUN: opt < %s -passes="loop-interchange"  -loop-interchange-undo-simple-reduction -pass-remarks-missed='loop-interchange' \
; RUN:            -pass-remarks-output=%t -S | FileCheck -check-prefix=IR %s
; RUN: FileCheck --input-file=%t %s


; 1. The initial value of the reduction is not a constant.
; for (int i = 0; i < n; i++) {
;   for (int j = 0; j < n; j++)
;     s[i] = s[i] + a[j][i] * b[j][i];
; }

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIInner
; CHECK-NEXT: Function:        simple_reduction_01
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Only inner loops with induction or reduction PHI nodes can be interchange currently.

; IR-LABEL: @simple_reduction_01(
; IR-NOT: split
define void @simple_reduction_01(ptr noalias noundef readonly captures(none) %a, ptr noalias noundef readonly captures(none) %b, ptr noalias noundef writeonly captures(none) %s, i64  noundef %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %outerloop_header, label %exit

outerloop_header:
  %index_i = phi i64 [ 0, %entry ], [ %index_i.next, %outerloop_latch ]
  %addr_s = getelementptr inbounds nuw double, ptr %s, i64 %index_i
  %invariant.gep.us = getelementptr inbounds nuw [100 x double], ptr %a, i64 0, i64 %index_i
  %invariant.gep32.us = getelementptr inbounds nuw [100 x double], ptr %b, i64 0, i64 %index_i
  %s_init = load double, ptr %addr_s, align 8
  br label %innerloop

innerloop:
  %index_j = phi i64 [ 0, %outerloop_header ], [ %index_j.next, %innerloop ]
  %reduction = phi double [ %s_init, %outerloop_header ], [ %add, %innerloop ]
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

; 2. There are two or more reductions
; for (int i = 0; i < n; i++) {
;   s[i] = 0;
;   s2[i] = 0;
;   for (int j = 0; j < n; j++){
;     s[i] = s[i] + a[j][i] * b[j][i];
;     s2[i] = s2[i] + a[j][i];
;   }
; }

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIInner
; CHECK-NEXT: Function:        simple_reduction_02
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Only inner loops with induction or reduction PHI nodes can be interchange currently.

; IR-LABEL: @simple_reduction_02(
; IR-NOT: split
define void @simple_reduction_02(ptr noalias noundef readonly captures(none) %a, ptr noalias noundef readonly captures(none) %b, ptr noalias noundef writeonly captures(none) %s, ptr noalias noundef writeonly captures(none) %s2, i64  noundef %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %outerloop_header, label %exit

outerloop_header:
  %index_i = phi i64 [ 0, %entry ], [ %index_i.next, %outerloop_latch ]
  %addr_s = getelementptr inbounds nuw double, ptr %s, i64 %index_i
  %addr_s2 = getelementptr inbounds nuw double, ptr %s2, i64 %index_i
  %invariant.gep.us = getelementptr inbounds nuw [100 x double], ptr %a, i64 0, i64 %index_i
  %invariant.gep32.us = getelementptr inbounds nuw [100 x double], ptr %b, i64 0, i64 %index_i
  br label %innerloop

innerloop:
  %index_j = phi i64 [ 0, %outerloop_header ], [ %index_j.next, %innerloop ]
  %reduction = phi double [ 0.000000e+00, %outerloop_header ], [ %add, %innerloop ]
  %reduction2 = phi double [ 0.000000e+00, %outerloop_header ], [ %add, %innerloop ]
  %addr_a_j_i = getelementptr inbounds nuw [100 x double], ptr %invariant.gep.us, i64 %index_j
  %0 = load double, ptr %addr_a_j_i, align 8
  %addr_b_j_i = getelementptr inbounds nuw [100 x double], ptr %invariant.gep32.us, i64 %index_j
  %1 = load double, ptr %addr_b_j_i, align 8
  %mul = fmul fast double %1, %0
  %add = fadd fast double %mul, %reduction
  %add2 = fadd fast double %reduction2, %0
  %index_j.next = add nuw nsw i64 %index_j, 1
  %cond1 = icmp eq i64 %index_j.next, %n
  br i1 %cond1, label %outerloop_latch, label %innerloop

outerloop_latch:
  %lcssa = phi double [ %add, %innerloop ]
  %lcssa2 = phi double [%add2, %innerloop]
  store double %lcssa, ptr %addr_s, align 8
  store double %lcssa2, ptr %addr_s2, align 8
  %index_i.next = add nuw nsw i64 %index_i, 1
  %cond2 = icmp eq i64 %index_i.next, %n
  br i1 %cond2, label %exit, label %outerloop_header

exit:
  ret void
}

; 3. The reduction is used more than twice in the outer loop.
; for (int i = 0; i < n; i++) {
;   s[i] = 0;
;   for (int j = 0; j < n; j++)
;     s[i] = s[i] + a[j][i] * b[j][i];
;   s[i] += 1;
; }

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIInner
; CHECK-NEXT: Function:        simple_reduction_03
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Only inner loops with induction or reduction PHI nodes can be interchange currently.

; IR-LABEL: @simple_reduction_03(
; IR-NOT: split
define void @simple_reduction_03(ptr noalias noundef readonly captures(none) %a, ptr noalias noundef readonly captures(none) %b, ptr noalias noundef writeonly captures(none) %s, i64  noundef %n) {
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
  %add17.us = fadd fast double %lcssa, 1.000000e+00
  store double %add17.us, ptr %addr_s, align 8
  %index_i.next = add nuw nsw i64 %index_i, 1
  %cond2 = icmp eq i64 %index_i.next, %n
  br i1 %cond2, label %exit, label %outerloop_header

exit:
  ret void
}


; 4. The reduction is not in the innermost loop.
; for (int i = 0; i < n; i++) {
;   s[i] = 0;
;   for (int j = 0; j < n; j++) {
;     s[i] = s[i] + a[j][i] * b[j][i]; // reduction
;     for (int k = 0; k < n; k++)
;       c[k] = 1;

;   }
; }

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIOuter
; CHECK-NEXT: Function:        simple_reduction_04
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Only outer loops with induction or reduction PHI nodes can be interchanged currently.

; IR-LABEL: @simple_reduction_04(
; IR-NOT: split
define void @simple_reduction_04(ptr noalias noundef readonly captures(none) %a, ptr noalias noundef readonly captures(none) %b, ptr noalias noundef writeonly captures(none) %c, ptr noalias noundef writeonly captures(none) %s, i64  noundef %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %i_loop_header, label %exit

i_loop_header:
  %index_i = phi i64 [ 0, %entry ], [ %index_i.next, %i_loop_latch ]
  %addr_s = getelementptr inbounds nuw double, ptr %s, i64 %index_i
  %invariant.gep.us = getelementptr inbounds nuw [100 x double], ptr %a, i64 0, i64 %index_i
  %invariant.gep32.us = getelementptr inbounds nuw [100 x double], ptr %b, i64 0, i64 %index_i
  br label %j_loop

j_loop:
  %index_j = phi i64 [ 0, %i_loop_header ], [ %index_j.next, %j_loop_latch ]
  %reduction = phi double [ 0.000000e+00, %i_loop_header ], [ %add, %j_loop_latch ]
  %addr_a_j_i = getelementptr inbounds nuw [100 x double], ptr %invariant.gep.us, i64 %index_j
  %0 = load double, ptr %addr_a_j_i, align 8
  %addr_b_j_i = getelementptr inbounds nuw [100 x double], ptr %invariant.gep32.us, i64 %index_j
  %1 = load double, ptr %addr_b_j_i, align 8
  %mul = fmul fast double %1, %0
  %add = fadd fast double %mul, %reduction
  br label %k_loop
  
k_loop:                                 
  %index_k = phi i64 [ %index_k.next, %k_loop ], [ 0, %j_loop ]
  %arrayidx22.us.us = getelementptr inbounds nuw double, ptr %c, i64 %index_k
  ; store double 1.000000e+00, ptr %arrayidx22.us.us, align 8 // Avoid unrelated store instructions from affecting the interchange of the i-loop and j-loop
  %index_k.next = add nuw nsw i64 %index_k, 1
  %exitcond.not = icmp eq i64 %index_k.next, %n
  br i1 %exitcond.not, label %j_loop_latch, label %k_loop

j_loop_latch:    
  %index_j.next = add nuw nsw i64 %index_j, 1
  %cond1 = icmp eq i64 %index_j.next, %n
  br i1 %cond1, label %i_loop_latch, label %j_loop

i_loop_latch:
  %lcssa = phi double [ %add, %j_loop_latch ]
  store double %lcssa, ptr %addr_s, align 8
  %index_i.next = add nuw nsw i64 %index_i, 1
  %cond2 = icmp eq i64 %index_i.next, %n
  br i1 %cond2, label %exit, label %i_loop_header

exit:
  ret void
}
