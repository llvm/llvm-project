; RUN: llc -O2 -march=bpfel -mcpu=v2 -mattr=+alu32 < %s | FileCheck %s
;
; For the below example, two phi node in the loop may depend on
; each other. So implementation must handle recursion properly.
;
; int test(unsigned long a, unsigned long b, unsigned long c) {
;   int val = 0;
;
;   #pragma clang loop unroll(disable)
;   for (long i = 0; i < 100; i++) {
;     if (a > b)
;       val = 1;
;     a += b;
;     if (b > c)
;       val = 1;
;     b += c;
;   }
;
;   return val == 0 ? 1 : 0;
; }


define dso_local i32 @test(i64 %a, i64 %b, i64 %c) local_unnamed_addr {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %cmp6 = icmp eq i32 %val.2, 0
  %cond = zext i1 %cmp6 to i32
  ret i32 %cond

for.body:                                         ; preds = %for.body, %entry
  %i.018 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %val.017 = phi i32 [ 0, %entry ], [ %val.2, %for.body ]
  %a.addr.016 = phi i64 [ %a, %entry ], [ %add, %for.body ]
  %b.addr.015 = phi i64 [ %b, %entry ], [ %add5, %for.body ]
  %cmp1 = icmp ugt i64 %a.addr.016, %b.addr.015
  %add = add i64 %a.addr.016, %b.addr.015
  %cmp2 = icmp ugt i64 %b.addr.015, %c
  %0 = or i1 %cmp2, %cmp1
  %val.2 = select i1 %0, i32 1, i32 %val.017
  %add5 = add i64 %b.addr.015, %c
  %inc = add nuw nsw i64 %i.018, 1
  %exitcond = icmp eq i64 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !2
}
; CHECK: [[VAL:r[0-9]+]] <<= 32
; CHECK: [[VAL]] >>= 32
; CHECK: if [[VAL]] == 0 goto

!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.disable"}
