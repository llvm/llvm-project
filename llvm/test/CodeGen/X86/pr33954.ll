; RUN: llc -mtriple=x86_64-pc-linux -x86-cmov-converter=true -verify-machineinstrs < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; This test checks that x86-cmov-converter optimization does not transform CMOV
;; instruction when the gain (in cycles) of converting to branch is less than
;; a fix threshold (measured for "-x86-cmov-converter-threshold=4").
;;
;; Test was created using the following command line:
;; > clang -S -O2 -m64 -fno-vectorize -fno-unroll-loops -emit-llvm foo.c -o -
;; Where foo.c is:
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;int bar(int *a, int *b, int n) {
;;  int sum = 0;
;;  for (int i = 0; i < n; ++i) {
;;    int x = a[i] * a[i+1] * a[i+2];
;;    int y = b[i] * b[i+1];
;;    sum += y > x ? x : 0;
;;  }
;;  return sum;
;;}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Adding a test to the above function shows code with CMOV is 25% faster than
;; the code with branch.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;#define N 10000
;;int A[N];
;;int B[N];
;;
;;
;;
;;int main () {
;;  for (int i=0; i< N; ++i) {
;;    A[i] = i%4;
;;    B[i] = i%5;
;;  }
;;  int sum = 0;
;;  for (int i=0; i< N*10; ++i)
;;    sum += bar(A, B, N);
;;  return sum;
;;}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-NOT: jg
; CHECK: cmovle
define i32 @bar(i32* nocapture readonly %a, i32* nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp30 = icmp sgt i32 %n, 0
  br i1 %cmp30, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %.pre = load i32, i32* %a, align 4
  %arrayidx2.phi.trans.insert = getelementptr inbounds i32, i32* %a, i64 1
  %.pre34 = load i32, i32* %arrayidx2.phi.trans.insert, align 4
  %.pre35 = load i32, i32* %b, align 4
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add14, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %for.body, %for.body.preheader
  %0 = phi i32 [ %.pre35, %for.body.preheader ], [ %5, %for.body ]
  %1 = phi i32 [ %.pre34, %for.body.preheader ], [ %4, %for.body ]
  %2 = phi i32 [ %.pre, %for.body.preheader ], [ %1, %for.body ]
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %sum.032 = phi i32 [ 0, %for.body.preheader ], [ %add14, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %mul = mul nsw i32 %1, %2
  %3 = add nuw nsw i64 %indvars.iv, 2
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %3
  %4 = load i32, i32* %arrayidx5, align 4
  %mul6 = mul nsw i32 %mul, %4
  %arrayidx11 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv.next
  %5 = load i32, i32* %arrayidx11, align 4
  %mul12 = mul nsw i32 %5, %0
  %cmp13 = icmp sgt i32 %mul12, %mul6
  %cond = select i1 %cmp13, i32 %mul6, i32 0
  %add14 = add nsw i32 %cond, %sum.032
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = {"target-cpu"="skylake"}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 5.0.0 (trunk)"}
