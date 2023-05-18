; RUN: opt %loadPolly -polly-import-jscop \
; RUN:     -polly-codegen -S < %s | FileCheck %s
;
;    void non-affine-update(double A[], double C[], double B[]) {
;      for (int i = 0; i < 10; i++) {
;        if (A[i] >= 6)
;          B[i] += 42;
;        else
;          C[i] += 3;
;      }
;    }

; Verify that all changed memory access functions are correctly code generated.
; At some point this did not work due to memory access identifiers not being
; unique within non-affine scop statements.

; CHECK: polly.stmt.bb2:
; CHECK:   %[[offset:.*]] = shl nuw nsw i64 %polly.indvar, 3
; CHECK:   %scevgep = getelementptr i8, ptr %A, i64 %[[offset]]

; CHECK: polly.stmt.bb9:
; CHECK:   %polly.access.C{{.*}} = getelementptr double, ptr %C, i64 42
; CHECK:   %polly.access.C{{.*}} = getelementptr double, ptr %C, i64 42

; CHECK: polly.stmt.bb5:
; CHECK:   %polly.access.B{{.*}} = getelementptr double, ptr %B, i64 113
; CHECK:   %polly.access.B{{.*}} = getelementptr double, ptr %B, i64 113


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @non-affine-update(ptr %A, ptr %C, ptr %B) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb14, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb14 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 10
  br i1 %exitcond, label %bb2, label %bb15

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds double, ptr %A, i64 %indvars.iv
  %tmp3 = load double, ptr %tmp, align 8
  %tmp4 = fcmp ult double %tmp3, 6.000000e+00
  br i1 %tmp4, label %bb9, label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  %tmp7 = load double, ptr %tmp6, align 8
  %tmp8 = fadd double %tmp7, 4.200000e+01
  store double %tmp8, ptr %tmp6, align 8
  br label %bb13

bb9:                                              ; preds = %bb2
  %tmp10 = getelementptr inbounds double, ptr %C, i64 %indvars.iv
  %tmp11 = load double, ptr %tmp10, align 8
  %tmp12 = fadd double %tmp11, 3.000000e+00
  store double %tmp12, ptr %tmp10, align 8
  br label %bb13

bb13:                                             ; preds = %bb9, %bb5
  br label %bb14

bb14:                                             ; preds = %bb13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb15:                                             ; preds = %bb1
  ret void
}
