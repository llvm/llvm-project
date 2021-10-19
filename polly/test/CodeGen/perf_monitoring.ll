; RUN: opt %loadPolly -polly-codegen -polly-codegen-perf-monitoring \
; RUN:   -S < %s | FileCheck %s

; void f(long A[], long N) {
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

; CHECK:      @__polly_perf_cycles_total_start = weak thread_local(initialexec) constant i64 0
; CHECK-NEXT: @__polly_perf_initialized = weak thread_local(initialexec) constant i1 false
; CHECK-NEXT: @__polly_perf_cycles_in_scops = weak thread_local(initialexec) constant i64 0
; CHECK-NEXT: @__polly_perf_cycles_in_scop_start = weak thread_local(initialexec) constant i64 0

; CHECK:      polly.split_new_and_old:                          ; preds = %entry
; CHECK-NEXT:   %0 = call { i64, i32 } @llvm.x86.rdtscp()
; CHECK-NEXT:   %1 = extractvalue { i64, i32 } %0, 0
; CHECK-NEXT:   store volatile i64 %1, i64* @__polly_perf_cycles_in_scop_start

; CHECK:      polly.merge_new_and_old:                          ; preds = %polly.exiting, %return.region_exiting
; CHECK-NEXT:   %6 = load volatile i64, i64* @__polly_perf_cycles_in_scop_start
; CHECK-NEXT:   %7 = call { i64, i32 } @llvm.x86.rdtscp()
; CHECK-NEXT:   %8 = extractvalue { i64, i32 } %7, 0
; CHECK-NEXT:   %9 = sub i64 %8, %6
; CHECK-NEXT:   %10 = load volatile i64, i64* @__polly_perf_cycles_in_scops
; CHECK-NEXT:   %11 = add i64 %10, %9
; CHECK-NEXT:   store volatile i64 %11, i64* @__polly_perf_cycles_in_scops


; CHECK:      define weak_odr void @__polly_perf_final() {
; CHECK-NEXT: start:
; CHECK-NEXT:   %0 = call { i64, i32 } @llvm.x86.rdtscp()
; CHECK-NEXT:   %1 = extractvalue { i64, i32 } %0, 0
; CHECK-NEXT:   %2 = load volatile i64, i64* @__polly_perf_cycles_total_start
; CHECK-NEXT:   %3 = sub i64 %1, %2
; CHECK-NEXT:   %4 = load volatile i64, i64* @__polly_perf_cycles_in_scops
; CHECK-NEXT:   %5 = call i32 (...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @1, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([27 x i8], [27 x i8] addrspace(4)* @0, i32 0, i32 0))
; CHECK-NEXT:   %6 = call i32 @fflush(i8* null)
; CHECK-NEXT:   %7 = call i32 (...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @3, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([27 x i8], [27 x i8] addrspace(4)* @2, i32 0, i32 0))
; CHECK-NEXT:   %8 = call i32 @fflush(i8* null)
; CHECK-NEXT:   %9 = call i32 (...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @6, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([8 x i8], [8 x i8] addrspace(4)* @4, i32 0, i32 0), i64 %3, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @5, i32 0, i32 0))
; CHECK-NEXT:   %10 = call i32 @fflush(i8* null)
; CHECK-NEXT:   %11 = call i32 (...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @9, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([8 x i8], [8 x i8] addrspace(4)* @7, i32 0, i32 0), i64 %4, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @8, i32 0, i32 0))
; CHECK-NEXT:   %12 = call i32 @fflush(i8* null)


; CHECK:      define weak_odr void @__polly_perf_init() {
; CHECK-NEXT: start:
; CHECK-NEXT:   %0 = load i1, i1* @__polly_perf_initialized
; CHECK-NEXT:   br i1 %0, label %earlyreturn, label %initbb

; CHECK:      earlyreturn:                                      ; preds = %start
; CHECK-NEXT:   ret void

; CHECK:      initbb:                                           ; preds = %start
; CHECK-NEXT:   store i1 true, i1* @__polly_perf_initialized
; CHECK-NEXT:   %1 = call i32 @atexit(i8* bitcast (void ()* @__polly_perf_final to i8*))
; CHECK-NEXT:   %2 = call { i64, i32 } @llvm.x86.rdtscp()
; CHECK-NEXT:   %3 = extractvalue { i64, i32 } %2, 0
; CHECK-NEXT:   store volatile i64 %3, i64* @__polly_perf_cycles_total_start
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
