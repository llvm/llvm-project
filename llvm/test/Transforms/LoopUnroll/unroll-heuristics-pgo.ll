; RUN: opt < %s -S -passes=loop-unroll -unroll-runtime -unroll-threshold=40 -unroll-max-percent-threshold-boost=100 | FileCheck %s

@known_constant = internal unnamed_addr constant [9 x i32] [i32 0, i32 -1, i32 0, i32 -1, i32 5, i32 -1, i32 0, i32 -1, i32 0], align 16

; CHECK-LABEL: @bar_prof
;       CHECK: entry:
;       CHECK:   br i1 %{{.*}}, label %[[LOOP_EPIL_PREHEADER:.*]], label %[[ENTRY_NEW:.*]], !prof ![[#PROF_UR_GUARD:]]
;       CHECK: [[ENTRY_NEW]]:
;       CHECK:   br label %loop
;       CHECK: loop:
;       CHECK:   %mul = mul
;       CHECK:   %mul.1 = mul
;       CHECK:   %mul.2 = mul
;       CHECK:   %mul.3 = mul
;       CHECK:   %mul.4 = mul
;       CHECK:   %mul.5 = mul
;       CHECK:   %mul.6 = mul
;       CHECK:   %mul.7 = mul
;   CHECK-NOT:   %mul.8 = mul
;       CHECK:   br i1 %{{.*}}, label %[[LOOP_END_UNR_LCSSA:.*]], label %loop, !prof ![[#PROF_UR_LATCH:]], !llvm.loop ![[#LOOP_UR_LATCH:]]
;       CHECK: [[LOOP_END_UNR_LCSSA]]:
;       CHECK:   br i1 %{{.*}}, label %[[LOOP_EPIL_PREHEADER]], label %loop.end, !prof ![[#PROF_RM_GUARD:]]
;       CHECK: [[LOOP_EPIL_PREHEADER]]:
;       CHECK:   br label %[[LOOP_EPIL:.*]]
;       CHECK: [[LOOP_EPIL]]:
;       CHECK:   br i1 %{{.*}}, label %[[LOOP_EPIL]], label %[[LOOP_END_EPILOG_LCSSA:.*]], !prof ![[#PROF_RM_LATCH:]], !llvm.loop ![[#LOOP_RM_LATCH:]]
define i32 @bar_prof(ptr noalias nocapture readonly %src, i64 %c) !prof !1 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %r  = phi i32 [ 0, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %src_element = load i32, ptr %arrayidx, align 4
  %array_const_idx = getelementptr inbounds [9 x i32], ptr @known_constant, i64 0, i64 %iv
  %const_array_element = load i32, ptr %array_const_idx, align 4
  %mul = mul nsw i32 %src_element, %const_array_element
  %add = add nsw i32 %mul, %r
  %inc = add nuw nsw i64 %iv, 1
  %exitcond86.i = icmp eq i64 %inc, %c
  br i1 %exitcond86.i, label %loop.end, label %loop, !prof !2

loop.end:
  %r.lcssa = phi i32 [ %r, %loop ]
  ret i32 %r.lcssa
}

; CHECK-LABEL: @bar_prof_flat
; CHECK-NOT: loop.epil
define i32 @bar_prof_flat(ptr noalias nocapture readonly %src, i64 %c) !prof !1 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %r  = phi i32 [ 0, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %src_element = load i32, ptr %arrayidx, align 4
  %array_const_idx = getelementptr inbounds [9 x i32], ptr @known_constant, i64 0, i64 %iv
  %const_array_element = load i32, ptr %array_const_idx, align 4
  %mul = mul nsw i32 %src_element, %const_array_element
  %add = add nsw i32 %mul, %r
  %inc = add nuw nsw i64 %iv, 1
  %exitcond86.i = icmp eq i64 %inc, %c
  br i1 %exitcond86.i, label %loop, label %loop.end, !prof !2

loop.end:
  %r.lcssa = phi i32 [ %r, %loop ]
  ret i32 %r.lcssa
}

!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 1, i32 1000}

; Original loop probability: p = 1000/(1+1000) = 0.999000999
; Original estimated trip count: (1+1000)/1 = 1001
; Unroll count: 8

; Probability of >=7 iterations after first: p^7 = 0.993027916 =~
; 2132511214 / (14972434 + 2132511214).
; CHECK: ![[#PROF_UR_GUARD]] = !{!"branch_weights", i32 14972434, i32 2132511214}

; Probability of >=8 more iterations: p^8 = 0.99203588 =~
; 2130380833 / (17102815 + 2130380833).
; CHECK: ![[#PROF_UR_LATCH]] = !{!"branch_weights", i32 17102815, i32 2130380833}

; 1001//8 = 125
; CHECK: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]]}
; CHECK: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 125}

; Probability of 1 to 7 more of 7 more remainder iterations:
; (p-p^8)/(1-p^8) = 0.874562282 =~ 1878108210 / (1878108210 + 269375438).
; CHECK: ![[#PROF_RM_GUARD]] = !{!"branch_weights", i32 1878108210, i32 269375438}

; Frequency of first remainder iter:   r1 =                      1
; Frequency of second remainder iter:  r2 = r1*(p-p^7)/(1-p^7) = 0.856714143
; Frequency of third remainder iter:   r3 = r2*(p-p^6)/(1-p^6) = 0.713571429
; Frequency of fourth remainder iter:  r4 = r2*(p-p^5)/(1-p^5) = 0.570571715
; Frequency of fifth remainder iter:   r5 = r2*(p-p^4)/(1-p^4) = 0.427714858
; Frequency of sixth remainder iter:   r6 = r2*(p-p^3)/(1-p^3) = 0.285000715
; Frequency of seventh remainder iter: r7 = r2*(p-p^2)/(1-p^2) = 0.142429143
; Solve for loop probability that produces that frequency: f = 1/(1-p') =>
; p' = 1-1/f = 1-1/(r1+r2+r3+r4+r5+r6+r7) = 0.749749875 =~
; 1610075606 / (1610075606 + 537408042).
; CHECK: ![[#PROF_RM_LATCH]] = !{!"branch_weights", i32 1610075606, i32 537408042}

; Remainder estimated trip count: 1001%8 = 1
; CHECK: ![[#LOOP_RM_LATCH]] = distinct !{![[#LOOP_RM_LATCH]], ![[#LOOP_RM_TC:]], ![[#DISABLE:]]}
; CHECK: ![[#LOOP_RM_TC]] = !{!"llvm.loop.estimated_trip_count", i32 1}
