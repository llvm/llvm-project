; RUN: opt %loadPolly -polly-codegen -polly-parallel -S < %s | FileCheck %s --check-prefix=AUTO
; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-num-threads=1 -S < %s | FileCheck %s --check-prefix=ONE
; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-num-threads=4 -S < %s | FileCheck %s --check-prefix=FOUR

; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-omp-backend=LLVM -S < %s | FileCheck %s --check-prefix=LIBOMP-AUTO
; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-omp-backend=LLVM -polly-num-threads=1 -S < %s | FileCheck %s --check-prefix=LIBOMP-ONE
; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-omp-backend=LLVM -polly-num-threads=4 -S < %s | FileCheck %s --check-prefix=LIBOMP-FOUR

; Ensure that the provided thread numbers are forwarded to the OpenMP calls.
;
;    void storePosition(int *A) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
;          A[i + j * 1024] = 0;
;    }

; AUTO: call void @GOMP_parallel_loop_runtime_start(ptr @storePosition_polly_subfn, ptr %polly.par.userContext{{[0-9]*}}, i32 0, i64 0, i64 1024, i64 1)
; ONE: call void @GOMP_parallel_loop_runtime_start(ptr @storePosition_polly_subfn, ptr %polly.par.userContext{{[0-9]*}}, i32 1, i64 0, i64 1024, i64 1)
; FOUR: call void @GOMP_parallel_loop_runtime_start(ptr @storePosition_polly_subfn, ptr %polly.par.userContext{{[0-9]*}}, i32 4, i64 0, i64 1024, i64 1)

; In automatic mode, no threads are pushed explicitly.
; LIBOMP-AUTO-NOT: call void @__kmpc_push_num_threads
; LIBOMP-ONE: call void @__kmpc_push_num_threads(ptr @.loc.dummy{{[.0-9]*}}, i32 %{{[0-9]+}}, i32 1)
; LIBOMP-FOUR: call void @__kmpc_push_num_threads(ptr @.loc.dummy{{[.0-9]*}}, i32 %{{[0-9]+}}, i32 4)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @storePosition(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc4 ], [ 0, %entry ]
  %exitcond5 = icmp ne i64 %indvars.iv3, 1024
  br i1 %exitcond5, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %tmp = shl nsw i64 %indvars.iv, 10
  %tmp6 = add nsw i64 %indvars.iv3, %tmp
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %tmp6
  store i32 0, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret void
}
