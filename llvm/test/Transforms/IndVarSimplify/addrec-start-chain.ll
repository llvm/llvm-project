; RUN: opt -passes=indvars -disable-output -stats %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; A chain of loops, where the IV of each loop starts at the exit value of the
; IV of the previous loop. So the start value of each IV is an AddRec of the
; previous loop.
;
; Check how many predicates SCEV checks on the start values of AddRecs.

; CHECK: {{^ *}}128 scalar-evolution - Number of predicates checked on the start values of AddRecs

target datalayout = "e-m:e-i64:64-n32:64-S128"

define void @chain(ptr %arg, ptr %g) {
entry:
  br label %pre0
pre0:
  br label %loop0
loop0:
  %p0 = phi i64 [ 0, %pre0 ], [ %add0, %loop0 ]
  %q0 = phi i32 [ 0, %pre0 ], [ %addq0, %loop0 ]
  %add0 = add nuw nsw i64 %p0, 1
  %gep0 = getelementptr inbounds nuw i8, ptr %arg, i64 %p0
  store i8 0, ptr %gep0, align 1
  %addq0 = add nuw nsw i32 %q0, 1
  %ld0 = load i32, ptr %g, align 4
  %sh0 = lshr i32 %addq0, %ld0
  %c1_0 = icmp ne i32 %sh0, 0
  %c2_0 = icmp ugt i64 %p0, 17407
  %or0 = or i1 %c2_0, %c1_0
  br i1 %or0, label %exit0, label %loop0
exit0:
  %e0 = phi i64 [ %add0, %loop0 ]
  %ec0 = phi i1 [ %c2_0, %loop0 ]
  br i1 %ec0, label %end, label %pre1
pre1:
  br label %loop1
loop1:
  %p1 = phi i64 [ %e0, %pre1 ], [ %add1, %loop1 ]
  %q1 = phi i32 [ 0, %pre1 ], [ %addq1, %loop1 ]
  %add1 = add nuw nsw i64 %p1, 1
  %gep1 = getelementptr inbounds nuw i8, ptr %arg, i64 %p1
  store i8 1, ptr %gep1, align 1
  %addq1 = add nuw nsw i32 %q1, 1
  %ld1 = load i32, ptr %g, align 4
  %sh1 = lshr i32 %addq1, %ld1
  %c1_1 = icmp ne i32 %sh1, 0
  %c2_1 = icmp ugt i64 %p1, 17407
  %or1 = or i1 %c2_1, %c1_1
  br i1 %or1, label %exit1, label %loop1
exit1:
  %e1 = phi i64 [ %add1, %loop1 ]
  %ec1 = phi i1 [ %c2_1, %loop1 ]
  br i1 %ec1, label %end, label %pre2
pre2:
  br label %loop2
loop2:
  %p2 = phi i64 [ %e1, %pre2 ], [ %add2, %loop2 ]
  %q2 = phi i32 [ 0, %pre2 ], [ %addq2, %loop2 ]
  %add2 = add nuw nsw i64 %p2, 1
  %gep2 = getelementptr inbounds nuw i8, ptr %arg, i64 %p2
  store i8 2, ptr %gep2, align 1
  %addq2 = add nuw nsw i32 %q2, 1
  %ld2 = load i32, ptr %g, align 4
  %sh2 = lshr i32 %addq2, %ld2
  %c1_2 = icmp ne i32 %sh2, 0
  %c2_2 = icmp ugt i64 %p2, 17407
  %or2 = or i1 %c2_2, %c1_2
  br i1 %or2, label %exit2, label %loop2
exit2:
  %e2 = phi i64 [ %add2, %loop2 ]
  %ec2 = phi i1 [ %c2_2, %loop2 ]
  br i1 %ec2, label %end, label %pre3
pre3:
  br label %loop3
loop3:
  %p3 = phi i64 [ %e2, %pre3 ], [ %add3, %loop3 ]
  %q3 = phi i32 [ 0, %pre3 ], [ %addq3, %loop3 ]
  %add3 = add nuw nsw i64 %p3, 1
  %gep3 = getelementptr inbounds nuw i8, ptr %arg, i64 %p3
  store i8 3, ptr %gep3, align 1
  %addq3 = add nuw nsw i32 %q3, 1
  %ld3 = load i32, ptr %g, align 4
  %sh3 = lshr i32 %addq3, %ld3
  %c1_3 = icmp ne i32 %sh3, 0
  %c2_3 = icmp ugt i64 %p3, 17407
  %or3 = or i1 %c2_3, %c1_3
  br i1 %or3, label %exit3, label %loop3
exit3:
  %e3 = phi i64 [ %add3, %loop3 ]
  %ec3 = phi i1 [ %c2_3, %loop3 ]
  br i1 %ec3, label %end, label %pre4
pre4:
  br label %loop4
loop4:
  %p4 = phi i64 [ %e3, %pre4 ], [ %add4, %loop4 ]
  %q4 = phi i32 [ 0, %pre4 ], [ %addq4, %loop4 ]
  %add4 = add nuw nsw i64 %p4, 1
  %gep4 = getelementptr inbounds nuw i8, ptr %arg, i64 %p4
  store i8 4, ptr %gep4, align 1
  %addq4 = add nuw nsw i32 %q4, 1
  %ld4 = load i32, ptr %g, align 4
  %sh4 = lshr i32 %addq4, %ld4
  %c1_4 = icmp ne i32 %sh4, 0
  %c2_4 = icmp ugt i64 %p4, 17407
  %or4 = or i1 %c2_4, %c1_4
  br i1 %or4, label %exit4, label %loop4
exit4:
  %e4 = phi i64 [ %add4, %loop4 ]
  %ec4 = phi i1 [ %c2_4, %loop4 ]
  br i1 %ec4, label %end, label %pre5
pre5:
  br label %loop5
loop5:
  %p5 = phi i64 [ %e4, %pre5 ], [ %add5, %loop5 ]
  %q5 = phi i32 [ 0, %pre5 ], [ %addq5, %loop5 ]
  %add5 = add nuw nsw i64 %p5, 1
  %gep5 = getelementptr inbounds nuw i8, ptr %arg, i64 %p5
  store i8 5, ptr %gep5, align 1
  %addq5 = add nuw nsw i32 %q5, 1
  %ld5 = load i32, ptr %g, align 4
  %sh5 = lshr i32 %addq5, %ld5
  %c1_5 = icmp ne i32 %sh5, 0
  %c2_5 = icmp ugt i64 %p5, 17407
  %or5 = or i1 %c2_5, %c1_5
  br i1 %or5, label %exit5, label %loop5
exit5:
  %e5 = phi i64 [ %add5, %loop5 ]
  %ec5 = phi i1 [ %c2_5, %loop5 ]
  br i1 %ec5, label %end, label %pre6
pre6:
  br label %loop6
loop6:
  %p6 = phi i64 [ %e5, %pre6 ], [ %add6, %loop6 ]
  %q6 = phi i32 [ 0, %pre6 ], [ %addq6, %loop6 ]
  %add6 = add nuw nsw i64 %p6, 1
  %gep6 = getelementptr inbounds nuw i8, ptr %arg, i64 %p6
  store i8 6, ptr %gep6, align 1
  %addq6 = add nuw nsw i32 %q6, 1
  %ld6 = load i32, ptr %g, align 4
  %sh6 = lshr i32 %addq6, %ld6
  %c1_6 = icmp ne i32 %sh6, 0
  %c2_6 = icmp ugt i64 %p6, 17407
  %or6 = or i1 %c2_6, %c1_6
  br i1 %or6, label %exit6, label %loop6
exit6:
  %e6 = phi i64 [ %add6, %loop6 ]
  %ec6 = phi i1 [ %c2_6, %loop6 ]
  br i1 %ec6, label %end, label %pre7
pre7:
  br label %loop7
loop7:
  %p7 = phi i64 [ %e6, %pre7 ], [ %add7, %loop7 ]
  %q7 = phi i32 [ 0, %pre7 ], [ %addq7, %loop7 ]
  %add7 = add nuw nsw i64 %p7, 1
  %gep7 = getelementptr inbounds nuw i8, ptr %arg, i64 %p7
  store i8 7, ptr %gep7, align 1
  %addq7 = add nuw nsw i32 %q7, 1
  %ld7 = load i32, ptr %g, align 4
  %sh7 = lshr i32 %addq7, %ld7
  %c1_7 = icmp ne i32 %sh7, 0
  %c2_7 = icmp ugt i64 %p7, 17407
  %or7 = or i1 %c2_7, %c1_7
  br i1 %or7, label %exit7, label %loop7
exit7:
  %e7 = phi i64 [ %add7, %loop7 ]
  %ec7 = phi i1 [ %c2_7, %loop7 ]
  br i1 %ec7, label %end, label %pre8
pre8:
  br label %end
end:
  ret void
}
