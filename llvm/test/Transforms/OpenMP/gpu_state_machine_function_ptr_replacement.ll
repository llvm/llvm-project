; RUN: opt -S -passes=openmp-opt -openmp-ir-builder-optimistic-attributes -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s
; RUN: opt -S -passes=openmp-opt -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s

; fix it later
; XFAIL: *

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
;     unknown();
;     #pragma omp parallel
;     {}
;   }
; }

; Verify we replace the function pointer uses for the first and last outlined
; region (1 and 3) but not for the middle one (2) because it could be called from
; another kernel.

; CHECK-DAG: @__omp_outlined__1_wrapper.ID = private constant i8 undef
; CHECK-DAG: @__omp_outlined__2_wrapper.ID = private constant i8 undef

; CHECK-DAG:   icmp eq ptr %worker.work_fn, @__omp_outlined__1_wrapper.ID
; CHECK-DAG:   icmp eq ptr %worker.work_fn, @__omp_outlined__2_wrapper.ID


; CHECK-DAG:   call void @__kmpc_parallel_51(ptr @1, i32 %{{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_outlined__1, ptr @__omp_outlined__1_wrapper.ID, ptr %{{.*}}, i64 0)
; CHECK-DAG:   call void @__kmpc_parallel_51(ptr @1, i32 %{{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_outlined__2, ptr @__omp_outlined__2_wrapper.ID, ptr %{{.*}}, i64 0)
; CHECK-DAG:   call void @__kmpc_parallel_51(ptr @2, i32 %{{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_outlined__3, ptr @__omp_outlined__3_wrapper, ptr %{{.*}}, i64 0)


%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 2, i32 0, ptr @0 }, align 8
@__omp_offloading_10301_87b2c_foo_l7_kernel_environment = local_unnamed_addr constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 0, i8 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 }, ptr @1, ptr null }

define weak void @__omp_offloading_10301_87b2c_foo_l7() "kernel" {
entry:
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, ptr %.zero.addr, align 4
  %0 = call i32 @__kmpc_target_init(ptr @__omp_offloading_10301_87b2c_foo_l7_kernel_environment, ptr null)
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i32 %1, ptr %.threadid_temp., align 4
  call void @__omp_outlined__(ptr %.threadid_temp., ptr %.zero.addr)
  call void @__kmpc_target_deinit()
  ret void

worker.exit:                                      ; preds = %entry
  ret void
}

define weak i32 @__kmpc_target_init(ptr %0, ptr) {
  ret i32 0
}

declare void @unknown()

define internal void @__omp_outlined__(ptr noalias %.global_tid., ptr noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca ptr, align 8
  %.bound_tid..addr = alloca ptr, align 8
  %captured_vars_addrs = alloca [0 x ptr], align 8
  %captured_vars_addrs1 = alloca [0 x ptr], align 8
  store ptr %.global_tid., ptr %.global_tid..addr, align 8
  store ptr %.bound_tid., ptr %.bound_tid..addr, align 8
  %0 = load ptr, ptr %.global_tid..addr, align 8
  %1 = load i32, ptr %0, align 4
  call void @__kmpc_parallel_51(ptr @1, i32 %1, i32 1, i32 -1, i32 -1, ptr @__omp_outlined__1, ptr @__omp_outlined__1_wrapper, ptr %captured_vars_addrs, i64 0)
  call void @bar()
  call void @unknown()
  call void @__kmpc_parallel_51(ptr @1, i32 %1, i32 1, i32 -1, i32 -1, ptr @__omp_outlined__2, ptr @__omp_outlined__2_wrapper, ptr %captured_vars_addrs1, i64 0)
  ret void
}

define internal void @__omp_outlined__1(ptr noalias %.global_tid., ptr noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca ptr, align 8
  %.bound_tid..addr = alloca ptr, align 8
  store ptr %.global_tid., ptr %.global_tid..addr, align 8
  store ptr %.bound_tid., ptr %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__1_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca ptr, align 8
  store i32 0, ptr %.zero.addr, align 4
  store i16 %0, ptr %.addr, align 2
  store i32 %1, ptr %.addr1, align 4
  call void @__kmpc_get_shared_variables(ptr %global_args)
  call void @__omp_outlined__1(ptr %.addr1, ptr %.zero.addr)
  ret void
}

declare void @__kmpc_get_shared_variables(ptr)

declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64)

define hidden void @bar() {
entry:
  %captured_vars_addrs = alloca [0 x ptr], align 8
  %0 = call i32 @__kmpc_global_thread_num(ptr @2)
  call void @__kmpc_parallel_51(ptr @2, i32 %0, i32 1, i32 -1, i32 -1, ptr @__omp_outlined__3, ptr @__omp_outlined__3_wrapper, ptr %captured_vars_addrs, i64 0)
  ret void
}

define internal void @__omp_outlined__2(ptr noalias %.global_tid., ptr noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca ptr, align 8
  %.bound_tid..addr = alloca ptr, align 8
  store ptr %.global_tid., ptr %.global_tid..addr, align 8
  store ptr %.bound_tid., ptr %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__2_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca ptr, align 8
  store i32 0, ptr %.zero.addr, align 4
  store i16 %0, ptr %.addr, align 2
  store i32 %1, ptr %.addr1, align 4
  call void @__kmpc_get_shared_variables(ptr %global_args)
  call void @__omp_outlined__2(ptr %.addr1, ptr %.zero.addr)
  ret void
}

declare i32 @__kmpc_global_thread_num(ptr)

declare void @__kmpc_target_deinit()

define internal void @__omp_outlined__3(ptr noalias %.global_tid., ptr noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca ptr, align 8
  %.bound_tid..addr = alloca ptr, align 8
  store ptr %.global_tid., ptr %.global_tid..addr, align 8
  store ptr %.bound_tid., ptr %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__3_wrapper(i16 zeroext %0, i32 %1) {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca ptr, align 8
  store i32 0, ptr %.zero.addr, align 4
  store i16 %0, ptr %.addr, align 2
  store i32 %1, ptr %.addr1, align 4
  call void @__kmpc_get_shared_variables(ptr %global_args)
  call void @__omp_outlined__3(ptr %.addr1, ptr %.zero.addr)
  ret void
}

!omp_offload.info = !{!0}
!nvvm.annotations = !{!1}
!llvm.module.flags = !{!2, !3}

!0 = !{i32 0, i32 66305, i32 555956, !"foo", i32 7, i32 0}
!1 = !{ptr @__omp_offloading_10301_87b2c_foo_l7, !"kernel", i32 1}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 7, !"openmp-device", i32 50}
