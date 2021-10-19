; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN:   -da-disable-delinearization-checks | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN:   | FileCheck --check-prefix=LIN %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"

;;  for (long int i = 0; i < n; i++) {
;;    for (long int j = 0; j < n; j++) {
;;      for (long int k = 0; k < n; k++) {
;;        A[i][j][k] = i;
;;      }
;;      for (long int k = 0; k < n; k++) {
;;        *B++ = A[i + 3][j + 2][k + 1];

define void @p2(i64 %n, [100 x [100 x i64]]* %A, i64* %B) nounwind uwtable ssp {
entry:
  %cmp10 = icmp sgt i64 %n, 0
  br i1 %cmp10, label %for.cond1.preheader.preheader, label %for.end26

; CHECK-LABEL: p2
; CHECK: da analyze - none!
; CHECK: da analyze - flow [-3 -2]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - output [* * *]!

; LIN-LABEL: p2
; LIN: da analyze - output [* * *]!
; LIN: da analyze - flow [* *|<]!
; LIN: da analyze - confused!
; LIN: da analyze - input [* * *]!
; LIN: da analyze - confused!
; LIN: da analyze - output [* * *]!

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc24
  %B.addr.012 = phi i64* [ %B.addr.1.lcssa, %for.inc24 ], [ %B, %for.cond1.preheader.preheader ]
  %i.011 = phi i64 [ %inc25, %for.inc24 ], [ 0, %for.cond1.preheader.preheader ]
  %cmp26 = icmp sgt i64 %n, 0
  br i1 %cmp26, label %for.cond4.preheader.preheader, label %for.inc24

for.cond4.preheader.preheader:                    ; preds = %for.cond1.preheader
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond4.preheader.preheader, %for.inc21
  %B.addr.18 = phi i64* [ %B.addr.2.lcssa, %for.inc21 ], [ %B.addr.012, %for.cond4.preheader.preheader ]
  %j.07 = phi i64 [ %inc22, %for.inc21 ], [ 0, %for.cond4.preheader.preheader ]
  %cmp51 = icmp sgt i64 %n, 0
  br i1 %cmp51, label %for.body6.preheader, label %for.cond10.loopexit

for.body6.preheader:                              ; preds = %for.cond4.preheader
  br label %for.body6

for.body6:                                        ; preds = %for.body6.preheader, %for.body6
  %k.02 = phi i64 [ %inc, %for.body6 ], [ 0, %for.body6.preheader ]
  %arrayidx8 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* %A, i64 %i.011, i64 %j.07, i64 %k.02
  store i64 %i.011, i64* %arrayidx8, align 8
  %inc = add nsw i64 %k.02, 1
  %exitcond13 = icmp ne i64 %inc, %n
  br i1 %exitcond13, label %for.body6, label %for.cond10.loopexit.loopexit

for.cond10.loopexit.loopexit:                     ; preds = %for.body6
  br label %for.cond10.loopexit

for.cond10.loopexit:                              ; preds = %for.cond10.loopexit.loopexit, %for.cond4.preheader
  %cmp113 = icmp sgt i64 %n, 0
  br i1 %cmp113, label %for.body12.preheader, label %for.inc21

for.body12.preheader:                             ; preds = %for.cond10.loopexit
  br label %for.body12

for.body12:                                       ; preds = %for.body12.preheader, %for.body12
  %k9.05 = phi i64 [ %inc19, %for.body12 ], [ 0, %for.body12.preheader ]
  %B.addr.24 = phi i64* [ %incdec.ptr, %for.body12 ], [ %B.addr.18, %for.body12.preheader ]
  %add = add nsw i64 %k9.05, 1
  %add13 = add nsw i64 %j.07, 2
  %add14 = add nsw i64 %i.011, 3
  %arrayidx17 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* %A, i64 %add14, i64 %add13, i64 %add
  %0 = load i64, i64* %arrayidx17, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.24, i64 1
  store i64 %0, i64* %B.addr.24, align 8
  %inc19 = add nsw i64 %k9.05, 1
  %exitcond = icmp ne i64 %inc19, %n
  br i1 %exitcond, label %for.body12, label %for.inc21.loopexit

for.inc21.loopexit:                               ; preds = %for.body12
  %scevgep = getelementptr i64, i64* %B.addr.18, i64 %n
  br label %for.inc21

for.inc21:                                        ; preds = %for.inc21.loopexit, %for.cond10.loopexit
  %B.addr.2.lcssa = phi i64* [ %B.addr.18, %for.cond10.loopexit ], [ %scevgep, %for.inc21.loopexit ]
  %inc22 = add nsw i64 %j.07, 1
  %exitcond14 = icmp ne i64 %inc22, %n
  br i1 %exitcond14, label %for.cond4.preheader, label %for.inc24.loopexit

for.inc24.loopexit:                               ; preds = %for.inc21
  %B.addr.2.lcssa.lcssa = phi i64* [ %B.addr.2.lcssa, %for.inc21 ]
  br label %for.inc24

for.inc24:                                        ; preds = %for.inc24.loopexit, %for.cond1.preheader
  %B.addr.1.lcssa = phi i64* [ %B.addr.012, %for.cond1.preheader ], [ %B.addr.2.lcssa.lcssa, %for.inc24.loopexit ]
  %inc25 = add nsw i64 %i.011, 1
  %exitcond15 = icmp ne i64 %inc25, %n
  br i1 %exitcond15, label %for.cond1.preheader, label %for.end26.loopexit

for.end26.loopexit:                               ; preds = %for.inc24
  br label %for.end26

for.end26:                                        ; preds = %for.end26.loopexit, %entry
  ret void
}
