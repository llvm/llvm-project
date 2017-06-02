; RUN: opt %loadPolly -polly-codegen -polly-codegen-perf-monitoring \
; RUN:   -S < %s | FileCheck %s

; void f(long A[], long N) {
;   long i;
;   if (true)
;     for (i = 0; i < N; ++i)
;       A[i] = i;
; }
; void g(long A[], long N) {
;   long i;
;   if (true)
;     for (i = 0; i < N; ++i)
;       A[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br label %next

next:
  br i1 true, label %for.i, label %return

for.i:
  %indvar = phi i64 [ 0, %next], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, i64* %A, i64 %indvar
  store i64 %indvar, i64* %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}


define void @g(i64* %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br label %next

next:
  br i1 true, label %for.i, label %return

for.i:
  %indvar = phi i64 [ 0, %next], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, i64* %A, i64 %indvar
  store i64 %indvar, i64* %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}

; Declaration of globals
; CHECK: @"__polly_perf_cycles_in_f_from__%next__to__%polly.merge_new_and_old" = weak thread_local(initialexec) constant i64 0
; CHECK: @"__polly_perf_cycles_in_g_from__%next__to__%polly.merge_new_and_old" = weak thread_local(initialexec) constant i64 0

; Bumping up counter in f
; CHECK:      polly.merge_new_and_old:                          ; preds = %polly.exiting, %return.region_exiting
; CHECK-NEXT:   %5 = load volatile i64, i64* @__polly_perf_cycles_in_scop_start
; CHECK-NEXT:   %6 = call i64 @llvm.x86.rdtscp(i8* bitcast (i32* @__polly_perf_write_loation to i8*))
; CHECK-NEXT:   %7 = sub i64 %6, %5
; CHECK-NEXT:   %8 = load volatile i64, i64* @__polly_perf_cycles_in_scops
; CHECK-NEXT:   %9 = add i64 %8, %7
; CHECK-NEXT:   store volatile i64 %9, i64* @__polly_perf_cycles_in_scops
; CHECK-NEXT:   %10 = load volatile i64, i64* @"__polly_perf_cycles_in_f_from__%next__to__%polly.merge_new_and_old"
; CHECK-NEXT:   %11 = add i64 %10, %7
; CHECK-NEXT:   store volatile i64 %11, i64* @"__polly_perf_cycles_in_f_from__%next__to__%polly.merge_new_and_old"
; CHECK-NEXT:   br label %return

; Bumping up counter in g
; CHECK:       polly.merge_new_and_old:                          ; preds = %polly.exiting, %return.region_exiting
; CHECK-NEXT:   %5 = load volatile i64, i64* @__polly_perf_cycles_in_scop_start
; CHECK-NEXT:   %6 = call i64 @llvm.x86.rdtscp(i8* bitcast (i32* @__polly_perf_write_loation to i8*))
; CHECK-NEXT:   %7 = sub i64 %6, %5
; CHECK-NEXT:   %8 = load volatile i64, i64* @__polly_perf_cycles_in_scops
; CHECK-NEXT:   %9 = add i64 %8, %7
; CHECK-NEXT:   store volatile i64 %9, i64* @__polly_perf_cycles_in_scops
; CHECK-NEXT:   %10 = load volatile i64, i64* @"__polly_perf_cycles_in_g_from__%next__to__%polly.merge_new_and_old"
; CHECK-NEXT:   %11 = add i64 %10, %7
; CHECK-NEXT:   store volatile i64 %11, i64* @"__polly_perf_cycles_in_g_from__%next__to__%polly.merge_new_and_old"
; CHECK-NEXT:   br label %return

; Final reporting prints
; CHECK:        %20 = load volatile i64, i64* @"__polly_perf_cycles_in_f_from__%next__to__%polly.merge_new_and_old"
; CHECK-NEXT:   %21 = call i32 (...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @25, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @18, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @19, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(4)* @20, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @21, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([25 x i8], [25 x i8] addrspace(4)* @22, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @23, i32 0, i32 0), i64 %20, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @24, i32 0, i32 0))
; CHECK-NEXT:   %22 = call i32 @fflush(i8* null)
; CHECK-NEXT:   %23 = load volatile i64, i64* @"__polly_perf_cycles_in_g_from__%next__to__%polly.merge_new_and_old"
; CHECK-NEXT:   %24 = call i32 (...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @33, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @26, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @27, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(4)* @28, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @29, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([25 x i8], [25 x i8] addrspace(4)* @30, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([3 x i8], [3 x i8] addrspace(4)* @31, i32 0, i32 0), i64 %23, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @32, i32 0, i32 0))
