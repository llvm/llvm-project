; RUN: opt %loadPolly -polly-detect -stats -disable-output < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK-DAG:  4 polly-detect     - Maximal number of loops in scops (profitable scops only)
; CHECK-DAG:  4 polly-detect     - Maximal number of loops in scops
; CHECK-DAG: 11 polly-detect     - Number of loops in scops (profitable scops only)
; CHECK-DAG: 11 polly-detect     - Number of loops in scops
; CHECK-DAG: 11 polly-detect     - Number of total loops
; CHECK-DAG:  5 polly-detect     - Number of scops (profitable scops only)
; CHECK-DAG:  1 polly-detect     - Number of scops with maximal loop depth 4 (profitable scops only)
; CHECK-DAG:  2 polly-detect     - Number of scops with maximal loop depth 1 (profitable scops only)
; CHECK-DAG:  1 polly-detect     - Number of scops with maximal loop depth 3 (profitable scops only)
; CHECK-DAG:  1 polly-detect     - Number of scops with maximal loop depth 2 (profitable scops only)
; CHECK-DAG:  5 polly-detect     - Number of scops
; CHECK-DAG:  1 polly-detect     - Number of scops with maximal loop depth 4
; CHECK-DAG:  2 polly-detect     - Number of scops with maximal loop depth 1
; CHECK-DAG:  1 polly-detect     - Number of scops with maximal loop depth 3
; CHECK-DAG:  1 polly-detect     - Number of scops with maximal loop depth 2

;    void foo_1d(float *A) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += i;
;    }
;
;    void foo_2d(float *A) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          A[i + j] += i + j;
;    }
;
;    void foo_3d(float *A) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          for (long k = 0; k < 1024; k++)
;            A[i + j + k] += i + j + k;
;    }
;
;    void foo_4d(float *A) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          for (long k = 0; k < 1024; k++)
;            for (long l = 0; l < 1024; l++)
;              A[i + j + k + l] += i + j + k + l;
;    }
;
;    void foo_zero_iterations(float *S) {
;      for (long i = 0; i < 0; i++)
;        A[i] += i;
;    }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo_1d(float* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = sitofp i64 %i.0 to float
  %tmp3 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, %tmp
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}

define void @foo_2d(float* %A) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb14, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp15, %bb14 ]
  %exitcond1 = icmp ne i64 %i.0, 1024
  br i1 %exitcond1, label %bb3, label %bb16

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb11, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp12, %bb11 ]
  %exitcond = icmp ne i64 %j.0, 1024
  br i1 %exitcond, label %bb5, label %bb13

bb5:                                              ; preds = %bb4
  %tmp = add nuw nsw i64 %i.0, %j.0
  %tmp6 = sitofp i64 %tmp to float
  %tmp7 = add nuw nsw i64 %i.0, %j.0
  %tmp8 = getelementptr inbounds float, float* %A, i64 %tmp7
  %tmp9 = load float, float* %tmp8, align 4
  %tmp10 = fadd float %tmp9, %tmp6
  store float %tmp10, float* %tmp8, align 4
  br label %bb11

bb11:                                             ; preds = %bb5
  %tmp12 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb13:                                             ; preds = %bb4
  br label %bb14

bb14:                                             ; preds = %bb13
  %tmp15 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb16:                                             ; preds = %bb2
  ret void
}

define void @foo_3d(float* %A) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb22, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp23, %bb22 ]
  %exitcond2 = icmp ne i64 %i.0, 1024
  br i1 %exitcond2, label %bb4, label %bb24

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb19, %bb4
  %j.0 = phi i64 [ 0, %bb4 ], [ %tmp20, %bb19 ]
  %exitcond1 = icmp ne i64 %j.0, 1024
  br i1 %exitcond1, label %bb6, label %bb21

bb6:                                              ; preds = %bb5
  br label %bb7

bb7:                                              ; preds = %bb16, %bb6
  %k.0 = phi i64 [ 0, %bb6 ], [ %tmp17, %bb16 ]
  %exitcond = icmp ne i64 %k.0, 1024
  br i1 %exitcond, label %bb8, label %bb18

bb8:                                              ; preds = %bb7
  %tmp = add nuw nsw i64 %i.0, %j.0
  %tmp9 = add nuw nsw i64 %tmp, %k.0
  %tmp10 = sitofp i64 %tmp9 to float
  %tmp11 = add nuw nsw i64 %i.0, %j.0
  %tmp12 = add nuw nsw i64 %tmp11, %k.0
  %tmp13 = getelementptr inbounds float, float* %A, i64 %tmp12
  %tmp14 = load float, float* %tmp13, align 4
  %tmp15 = fadd float %tmp14, %tmp10
  store float %tmp15, float* %tmp13, align 4
  br label %bb16

bb16:                                             ; preds = %bb8
  %tmp17 = add nuw nsw i64 %k.0, 1
  br label %bb7

bb18:                                             ; preds = %bb7
  br label %bb19

bb19:                                             ; preds = %bb18
  %tmp20 = add nuw nsw i64 %j.0, 1
  br label %bb5

bb21:                                             ; preds = %bb5
  br label %bb22

bb22:                                             ; preds = %bb21
  %tmp23 = add nuw nsw i64 %i.0, 1
  br label %bb3

bb24:                                             ; preds = %bb3
  ret void
}

define void @foo_4d(float* %A) {
bb:
  br label %bb4

bb4:                                              ; preds = %bb30, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp31, %bb30 ]
  %exitcond3 = icmp ne i64 %i.0, 1024
  br i1 %exitcond3, label %bb5, label %bb32

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb27, %bb5
  %j.0 = phi i64 [ 0, %bb5 ], [ %tmp28, %bb27 ]
  %exitcond2 = icmp ne i64 %j.0, 1024
  br i1 %exitcond2, label %bb7, label %bb29

bb7:                                              ; preds = %bb6
  br label %bb8

bb8:                                              ; preds = %bb24, %bb7
  %k.0 = phi i64 [ 0, %bb7 ], [ %tmp25, %bb24 ]
  %exitcond1 = icmp ne i64 %k.0, 1024
  br i1 %exitcond1, label %bb9, label %bb26

bb9:                                              ; preds = %bb8
  br label %bb10

bb10:                                             ; preds = %bb21, %bb9
  %l.0 = phi i64 [ 0, %bb9 ], [ %tmp22, %bb21 ]
  %exitcond = icmp ne i64 %l.0, 1024
  br i1 %exitcond, label %bb11, label %bb23

bb11:                                             ; preds = %bb10
  %tmp = add nuw nsw i64 %i.0, %j.0
  %tmp12 = add nuw nsw i64 %tmp, %k.0
  %tmp13 = add nuw nsw i64 %tmp12, %l.0
  %tmp14 = sitofp i64 %tmp13 to float
  %tmp15 = add nuw nsw i64 %i.0, %j.0
  %tmp16 = add nuw nsw i64 %tmp15, %k.0
  %tmp17 = add nuw nsw i64 %tmp16, %l.0
  %tmp18 = getelementptr inbounds float, float* %A, i64 %tmp17
  %tmp19 = load float, float* %tmp18, align 4
  %tmp20 = fadd float %tmp19, %tmp14
  store float %tmp20, float* %tmp18, align 4
  br label %bb21

bb21:                                             ; preds = %bb11
  %tmp22 = add nuw nsw i64 %l.0, 1
  br label %bb10

bb23:                                             ; preds = %bb10
  br label %bb24

bb24:                                             ; preds = %bb23
  %tmp25 = add nuw nsw i64 %k.0, 1
  br label %bb8

bb26:                                             ; preds = %bb8
  br label %bb27

bb27:                                             ; preds = %bb26
  %tmp28 = add nuw nsw i64 %j.0, 1
  br label %bb6

bb29:                                             ; preds = %bb6
  br label %bb30

bb30:                                             ; preds = %bb29
  %tmp31 = add nuw nsw i64 %i.0, 1
  br label %bb4

bb32:                                             ; preds = %bb4
  ret void
}

define void @foo_zero_iterations(float* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 0
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = sitofp i64 %i.0 to float
  %tmp3 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, %tmp
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
