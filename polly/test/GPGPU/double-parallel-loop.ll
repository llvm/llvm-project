; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-schedule \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=SCHED %s

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; REQUIRES: pollyacc

; CHECK: Stmt_bb5
; CHECK:       Domain :=
; CHECK:           { Stmt_bb5[i0, i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 };
; CHECK:       Schedule :=
; CHECK:           { Stmt_bb5[i0, i1] -> [i0, i1] };
; CHECK:       ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:           { Stmt_bb5[i0, i1] -> MemRef_A[i0, i1] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:           { Stmt_bb5[i0, i1] -> MemRef_A[i0, i1] };

; SCHED: domain: "{ Stmt_bb5[i0, i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 }"
; SCHED: child:
; SCHED:   context: "{ [] }"
; SCHED:   child:
; SCHED:     extension: "{  }"
; SCHED:     child:
; SCHED:       sequence:
; SCHED:       - filter: "{  }"
; SCHED:       - filter: "{ Stmt_bb5[i0, i1] }"
; SCHED:         child:
; SCHED:           guard: "{ [] }"
; SCHED:           child:
; SCHED:             mark: "kernel"
; SCHED:             child:
; SCHED:               context: "[b0, b1, t0, t1] -> { [] : 0 <= b0 <= 31 and 0 <= b1 <= 31 and 0 <= t0 <= 31 and 0 <= t1 <= 15 }"
; SCHED:               child:
; SCHED:                 filter: "[b0, b1] -> { Stmt_bb5[i0, i1] : -31 - 32b0 + i0 <= 8192*floor((i0)/8192) <= -32b0 + i0 and -31 - 32b1 + i1 <= 8192*floor((i1)/8192) <= -32b1 + i1 }"
; SCHED:                 child:
; SCHED:                   schedule: "[{ Stmt_bb5[i0, i1] -> [(floor((i0)/8192))] }, { Stmt_bb5[i0, i1] -> [(floor((i1)/8192))] }]"
; SCHED:                   permutable: 1
; SCHED:                   coincident: [ 1, 1 ]
; SCHED:                   child:
; SCHED:                     filter: "[t0, t1] -> { Stmt_bb5[i0, i1] : 32*floor((-t0 + i0)/32) = -t0 + i0 and 16*floor((-t1 + i1)/16) = -t1 + i1 and 0 <= t0 <= 31 and 0 <= t1 <= 15 }"
; SCHED:                     child:
; SCHED:                       schedule: "[{ Stmt_bb5[i0, i1] -> [(0)] }, { Stmt_bb5[i0, i1] -> [(floor((i1)/16) - 2*floor((i1)/32))] }]"
; SCHED:                       permutable: 1
; SCHED:                       coincident: [ 1, 1 ]
; SCHED:       - filter: "{  }"

; CODE: Code
; CODE: ====
; CODE: # host
; CODE: {
; CODE:   dim3 k0_dimBlock(16, 32);
; CODE:   dim3 k0_dimGrid(32, 32);
; CODE:   kernel0 <<<k0_dimGrid, k0_dimBlock>>> ();
; CODE: }

; CODE: # kernel0
; CODE: for (int c3 = 0; c3 <= 1; c3 += 1)
; CODE:   Stmt_bb5(32 * b0 + t0, 32 * b1 + t1 + 16 * c3);


;    void double_parallel_loop(float A[][1024]) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          A[i][j] += i * j;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @double_parallel_loop([1024 x float]* %A) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb13, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp14, %bb13 ]
  %exitcond1 = icmp ne i64 %i.0, 1024
  br i1 %exitcond1, label %bb3, label %bb15

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb10, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp11, %bb10 ]
  %exitcond = icmp ne i64 %j.0, 1024
  br i1 %exitcond, label %bb5, label %bb12

bb5:                                              ; preds = %bb4
  %tmp = mul nuw nsw i64 %i.0, %j.0
  %tmp6 = sitofp i64 %tmp to float
  %tmp7 = getelementptr inbounds [1024 x float], [1024 x float]* %A, i64 %i.0, i64 %j.0
  %tmp8 = load float, float* %tmp7, align 4
  %tmp9 = fadd float %tmp8, %tmp6
  store float %tmp9, float* %tmp7, align 4
  br label %bb10

bb10:                                             ; preds = %bb5
  %tmp11 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb12:                                             ; preds = %bb4
  br label %bb13

bb13:                                             ; preds = %bb12
  %tmp14 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb15:                                             ; preds = %bb2
  ret void
}
