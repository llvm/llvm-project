; RUN: opt %s -passes=loop-interchange -cache-line-size=0 -pass-remarks-output=%t -verify-dom-info -verify-loop-info \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange -disable-output
; RUN: FileCheck -input-file %t %s

;; In the following code, interchanging is unprofitable even if the cache line
;; size is set to zero. There are cases where the default cache line size is
;; zero, e.g., the target processor is not specified.
;;
;; #define N 100
;; #define M 100
;; 
;; // Extracted from SingleSource/Benchmarks/Polybench/datamining/correlation/correlation.c
;; // in llvm-test-suite
;; void f(double data[N][M], double mean[M], double stddev[M]) {
;;   for (int i = 0; i < N; i++) {
;;     for (int j = 0; j < M; j++) {
;;       data[i][j] -= mean[j];
;;       data[i][j] /= stddev[j];
;;     }
;;   }
;; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK:      Name:            InterchangeNotProfitable
; CHECK-NEXT: Function:        f

define void @f(ptr noundef captures(none) %data, ptr noundef readonly captures(none) %mean, ptr noundef readonly captures(none) %stddev) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv30 = phi i64 [ 0, %entry ], [ %indvars.iv.next31, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond.cleanup3:
  %indvars.iv.next31 = add nuw nsw i64 %indvars.iv30, 1
  %exitcond33 = icmp ne i64 %indvars.iv.next31, 100
  br i1 %exitcond33, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx = getelementptr inbounds nuw double, ptr %mean, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %arrayidx8 = getelementptr inbounds nuw [100 x double], ptr %data, i64 %indvars.iv30, i64 %indvars.iv
  %1 = load double, ptr %arrayidx8, align 8
  %sub = fsub double %1, %0
  store double %sub, ptr %arrayidx8, align 8
  %arrayidx10 = getelementptr inbounds nuw double, ptr %stddev, i64 %indvars.iv
  %2 = load double, ptr %arrayidx10, align 8
  %div = fdiv double %sub, %2
  store double %div, ptr %arrayidx8, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3
}
