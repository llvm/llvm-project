; Remove 'S' Scalar Dependencies #119345
; Scalar dependencies are not handled correctly, so they were removed to avoid
; miscompiles. The loop nest in this test case used to be interchanged, but it's
; no longer triggering. XFAIL'ing this test to indicate that this test should
; interchanged if scalar deps are handled correctly.
;
; XFAIL: *

; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks-output=%t -verify-dom-info -verify-loop-info \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange
; RUN: FileCheck -input-file %t %s

; RUN: opt < %s -passes=loop-interchange,loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks-output=%t -pass-remarks='loop-interchange' -S
; RUN: cat %t |  FileCheck --check-prefix=PROFIT %s

;; We test profitability model in these test cases.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x [100 x i32]] zeroinitializer

;;---------------------------------------Test case 01---------------------------------
;; Loops interchange will result in better cache locality and hence profitable. Check for interchange.
;;   for(int i=1;i<100;i++)
;;     for(int j=1;j<100;j++)
;;       A[j][i] = A[j - 1][i] + B[j][i];

; CHECK:      Name:            Interchanged
; CHECK-NEXT: Function:        interchange_01

define void @interchange_01() {
entry:
  br label %for2.preheader

for2.preheader:
  %i30 = phi i64 [ 1, %entry ], [ %i.next31, %for1.inc14 ]
  br label %for2

for2:
  %j = phi i64 [ %i.next, %for2 ], [ 1, %for2.preheader ]
  %j.prev = add nsw i64 %j,  -1
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %j.prev, i64 %i30
  %lv1 = load i32, ptr %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], ptr @B, i64 0, i64 %j,  i64 %i30
  %lv2 = load i32, ptr %arrayidx9
  %add = add nsw i32 %lv1, %lv2
  %arrayidx13 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %j,  i64 %i30
  store i32 %add, ptr %arrayidx13
  %i.next = add nuw nsw i64 %j,  1
  %exitcond = icmp eq i64 %j,  99
  br i1 %exitcond, label %for1.inc14, label %for2

for1.inc14:
  %i.next31 = add nuw nsw i64 %i30, 1
  %exitcond33 = icmp eq i64 %i30, 99
  br i1 %exitcond33, label %for.end16, label %for2.preheader

for.end16:
  ret void
}

;; ---------------------------------------Test case 02---------------------------------
;; Check loop interchange profitability model.
;; This tests profitability model when operands of getelementpointer and not exactly the induction variable but some
;; arithmetic operation on them.
;;   for(int i=1;i<N;i++)
;;    for(int j=1;j<N;j++)
;;       A[j-1][i-1] = A[j - 1][i-1] + B[j-1][i-1];

; CHECK:      Name:            Interchanged
; CHECK-NEXT: Function:        interchange_02
define void @interchange_02() {
entry:
  br label %for1.header

for1.header:
  %i35 = phi i64 [ 1, %entry ], [ %i.next36, %for1.inc19 ]
  %i.prev = add nsw i64 %i35, -1
  br label %for2

for2:
  %j = phi i64 [ 1, %for1.header ], [ %i.next, %for2 ]
  %j.prev = add nsw i64 %j,  -1
  %arrayidx6 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %j.prev, i64 %i.prev
  %lv1 = load i32, ptr %arrayidx6
  %arrayidx12 = getelementptr inbounds [100 x [100 x i32]], ptr @B, i64 0, i64 %j.prev, i64 %i.prev
  %lv2 = load i32, ptr %arrayidx12
  %add = add nsw i32 %lv1, %lv2
  store i32 %add, ptr %arrayidx6
  %i.next = add nuw nsw i64 %j,  1
  %exitcond = icmp eq i64 %j,  99
  br i1 %exitcond, label %for1.inc19, label %for2

for1.inc19:
  %i.next36 = add nuw nsw i64 %i35, 1
  %exitcond39 = icmp eq i64 %i35, 99
  br i1 %exitcond39, label %for.end21, label %for1.header

for.end21:
  ret void
}

;;---------------------------------------Test case 03---------------------------------
;; Loops interchange is not profitable.
;;   for(int i=1;i<100;i++)
;;     for(int j=1;j<100;j++)
;;       A[i-1][j-1] = A[i - 1][j-1] + B[i][j];

; CHECK:      Name:            InterchangeNotProfitable
; CHECK-NEXT: Function:        interchange_03
define void @interchange_03(){
entry:
  br label %for1.header

for1.header:
  %i34 = phi i64 [ 1, %entry ], [ %i.next35, %for1.inc17 ]
  %i.prev = add nsw i64 %i34, -1
  br label %for2

for2:
  %j = phi i64 [ 1, %for1.header ], [ %i.next, %for2 ]
  %j.prev = add nsw i64 %j, -1
  %arrayidx6 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %i.prev, i64 %j.prev
  %lv1 = load i32, ptr %arrayidx6
  %arrayidx10 = getelementptr inbounds [100 x [100 x i32]], ptr @B, i64 0, i64 %i34, i64 %j
  %lv2 = load i32, ptr %arrayidx10
  %add = add nsw i32 %lv1, %lv2
  store i32 %add, ptr %arrayidx6
  %i.next = add nuw nsw i64 %j,  1
  %exitcond = icmp eq i64 %j,  99
  br i1 %exitcond, label %for1.inc17, label %for2

for1.inc17:
  %i.next35 = add nuw nsw i64 %i34, 1
  %exitcond38 = icmp eq i64 %i34, 99
  br i1 %exitcond38, label %for.end19, label %for1.header

for.end19:
  ret void
}

;; Loops should not be interchanged in this case as it is not profitable.
;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<100;j++)
;;      A[i][j] = A[i][j]+k;

; CHECK:      Name:            InterchangeNotProfitable
; CHECK-NEXT: Function:        interchange_04
define void @interchange_04(i32 %k) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc10 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %indvars.iv21, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx5
  %add = add nsw i32 %0, %k
  store i32 %add, ptr %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 100
  br i1 %exitcond23, label %for.end12, label %for.cond1.preheader

for.end12:
  ret void
}

;;---------------------------------------Test case 05---------------------------------
;; This test is to make sure, that multiple invocations of loop interchange will not
;; undo previous interchange and will converge to a particular order determined by the
;; profitability analysis.
;;   for(int i=1;i<100;i++)
;;     for(int j=1;j<100;j++)
;;       A[j][0] = A[j][0] + B[j][i];

; CHECK:      Name:            Interchanged
; CHECK-NEXT: Function:        interchange_05

; PROFIT-LABEL: --- !Passed
; PROFIT-NEXT: Pass:            loop-interchange
; PROFIT-NEXT: Name:            Interchanged
; PROFIT-LABEL: Function:        interchange_05
; PROFIT-NEXT: Args:
; PROFIT-NEXT:   - String:          Loop interchanged with enclosing loop.
; PROFIT-NEXT: ...
; PROFIT: --- !Missed
; PROFIT-NEXT: Pass:            loop-interchange
; PROFIT-NEXT: Name:            InterchangeNotProfitable
; PROFIT-NEXT: Function:        interchange_05
; PROFIT-NEXT: Args:
; PROFIT-NEXT:   - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
; PROFIT-NEXT: ...
define void @interchange_05() {
entry:
  br label %for2.preheader

for2.preheader:
  %i30 = phi i64 [ 1, %entry ], [ %i.next31, %for1.inc14 ]
  br label %for2

for2:
  %j = phi i64 [ %i.next, %for2 ], [ 1, %for2.preheader ]
  %j.prev = add nsw i64 %j,  -1
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %j, i64 0
  %lv1 = load i32, ptr %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], ptr @B, i64 0, i64 %j,  i64 %i30
  %lv2 = load i32, ptr %arrayidx9
  %add = add nsw i32 %lv1, %lv2
  %arrayidx13 = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %j,  i64 0
  store i32 %add, ptr %arrayidx13
  %i.next = add nuw nsw i64 %j,  1
  %exitcond = icmp eq i64 %j,  99
  br i1 %exitcond, label %for1.inc14, label %for2

for1.inc14:
  %i.next31 = add nuw nsw i64 %i30, 1
  %exitcond33 = icmp eq i64 %i30, 99
  br i1 %exitcond33, label %for.end16, label %for2.preheader

for.end16:
  ret void
}

