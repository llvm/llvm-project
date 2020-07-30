; RUN: opt -S -passes=openmpopt -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s
; RUN: opt -S        -openmpopt -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s

; C input used for this test:

; void bar(void) {
;     #pragma omp parallel
;     { }
; }
; void foo(void) {
;   #pragma omp target teams
;   {
;     #pragma omp parallel
;     {}
;     bar();
;     #pragma omp parallel
;     {}
;   }
; }

; Verify we replace the function pointer uses for the first and last outlined
; region (1 and 3) but not for the middle one (2) because it could be called from
; another kernel.

; CHECK-DAG: @__omp_outlined__1_wrapper.ID = private constant i8 undef
; CHECK-DAG: @__omp_outlined__3_wrapper.ID = private constant i8 undef

; CHECK-DAG:   icmp eq i8* %5, @__omp_outlined__1_wrapper.ID
; CHECK-DAG:   icmp eq i8* %7, @__omp_outlined__3_wrapper.ID

; CHECK-DAG:   call void @__kmpc_kernel_prepare_parallel(i8* @__omp_outlined__1_wrapper.ID)
; CHECK-DAG:   call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void ()* @__omp_outlined__2_wrapper to i8*))
; CHECK-DAG:   call void @__kmpc_kernel_prepare_parallel(i8* @__omp_outlined__3_wrapper.ID)


%struct.ident_t = type { i32, i32, i32, i32, i8* }

define internal void @__omp_offloading_35_a1e179_foo_l7_worker() {
entry:
  %work_fn = alloca i8*, align 8
  %exec_status = alloca i8, align 1
  store i8* null, i8** %work_fn, align 8
  store i8 0, i8* %exec_status, align 1
  br label %.await.work

.await.work:                                      ; preds = %.barrier.parallel, %entry
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  %0 = call i1 @__kmpc_kernel_parallel(i8** %work_fn)
  %1 = zext i1 %0 to i8
  store i8 %1, i8* %exec_status, align 1
  %2 = load i8*, i8** %work_fn, align 8
  %should_terminate = icmp eq i8* %2, null
  br i1 %should_terminate, label %.exit, label %.select.workers

.select.workers:                                  ; preds = %.await.work
  %3 = load i8, i8* %exec_status, align 1
  %is_active = icmp ne i8 %3, 0
  br i1 %is_active, label %.execute.parallel, label %.barrier.parallel

.execute.parallel:                                ; preds = %.select.workers
  %4 = call i32 @__kmpc_global_thread_num(%struct.ident_t* null)
  %5 = load i8*, i8** %work_fn, align 8
  %work_match = icmp eq i8* %5, bitcast (void ()* @__omp_outlined__1_wrapper to i8*)
  br i1 %work_match, label %.execute.fn, label %.check.next

.execute.fn:                                      ; preds = %.execute.parallel
  call void @__omp_outlined__1_wrapper()
  br label %.terminate.parallel

.check.next:                                      ; preds = %.execute.parallel
  %6 = load i8*, i8** %work_fn, align 8
  %work_match1 = icmp eq i8* %6, bitcast (void ()* @__omp_outlined__2_wrapper to i8*)
  br i1 %work_match1, label %.execute.fn2, label %.check.next3

.execute.fn2:                                     ; preds = %.check.next
  call void @__omp_outlined__2_wrapper()
  br label %.terminate.parallel

.check.next3:                                     ; preds = %.check.next
  %7 = load i8*, i8** %work_fn, align 8
  %work_match4 = icmp eq i8* %7, bitcast (void ()* @__omp_outlined__3_wrapper to i8*)
  br i1 %work_match4, label %.execute.fn5, label %.check.next6

.execute.fn5:                                     ; preds = %.check.next3
  call void @__omp_outlined__3_wrapper()
  br label %.terminate.parallel

.check.next6:                                     ; preds = %.check.next3
  %8 = bitcast i8* %2 to void ()*
  call void %8()
  br label %.terminate.parallel

.terminate.parallel:                              ; preds = %.check.next6, %.execute.fn5, %.execute.fn2, %.execute.fn
  call void @__kmpc_kernel_end_parallel()
  br label %.barrier.parallel

.barrier.parallel:                                ; preds = %.terminate.parallel, %.select.workers
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  br label %.await.work

.exit:                                            ; preds = %.await.work
  ret void
}

define weak void @__omp_offloading_35_a1e179_foo_l7() {
  call void @__omp_offloading_35_a1e179_foo_l7_worker()
  call void @__omp_outlined__()
  ret void
}

define internal void @__omp_outlined__() {
  call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void ()* @__omp_outlined__1_wrapper to i8*))
  call void @bar()
  call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void ()* @__omp_outlined__3_wrapper to i8*))
  ret void
}

define internal void @__omp_outlined__1() {
  ret void
}

define internal void @__omp_outlined__1_wrapper() {
  call void @__omp_outlined__1()
  ret void
}

define hidden void @bar() {
  call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void ()* @__omp_outlined__2_wrapper to i8*))
  ret void
}

define internal void @__omp_outlined__2_wrapper() {
  ret void
}

define internal void @__omp_outlined__3_wrapper() {
  ret void
}

declare void @__kmpc_kernel_prepare_parallel(i8* %WorkFn)

declare zeroext i1 @__kmpc_kernel_parallel(i8** nocapture %WorkFn)

declare void @__kmpc_kernel_end_parallel()

declare void @__kmpc_barrier_simple_spmd(%struct.ident_t* nocapture readnone %loc_ref, i32 %tid)

declare i32 @__kmpc_global_thread_num(%struct.ident_t* nocapture readnone)


!nvvm.annotations = !{!0}

!0 = !{void ()* @__omp_offloading_35_a1e179_foo_l7, !"kernel", i32 1}
