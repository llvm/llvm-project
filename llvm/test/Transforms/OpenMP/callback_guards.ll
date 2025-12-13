; RUN: opt -passes=openmp-opt -S < %s | FileCheck %s

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.DynamicEnvironmentTy = type { i16 }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }

@0 = private unnamed_addr addrspace(1) constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr addrspacecast (ptr addrspace(1) @0 to ptr) }, align 8
@__omp_offloading_10303_1849aab__QQmain_l22_exec_mode = weak protected addrspace(1) constant i8 1
@__omp_offloading_10303_1849aab__QQmain_l22_dynamic_environment = weak_odr protected addrspace(1) global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_10303_1849aab__QQmain_l22_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 256, i32 0, i32 0, i32 4, i32 1024 }, ptr addrspacecast (ptr addrspace(1) @1 to ptr), ptr addrspacecast (ptr addrspace(1) @__omp_offloading_10303_1849aab__QQmain_l22_dynamic_environment to ptr) }

; Function Attrs: nounwind
define internal void @parallel_func_..omp_par.3(ptr noalias noundef %tid.addr.ascast, ptr noalias noundef %zero.addr.ascast, ptr %0) #1 {
omp.par.entry:
  ret void
}

; Function Attrs: mustprogress
define weak_odr protected amdgpu_kernel void @__omp_offloading_10303_1849aab__QQmain_l22(ptr %0, ptr %1, ptr %2) #4 {
entry:
  %7 = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @__omp_offloading_10303_1849aab__QQmain_l22_kernel_environment to ptr), ptr %0)
  %exec_user_code = icmp eq i32 %7, -1
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit

user_code.entry:                                  ; preds = %entry
  call void @__kmpc_distribute_static_loop_4u(ptr addrspacecast (ptr addrspace(1) @1 to ptr), ptr @__omp_offloading_10303_1849aab__QQmain_l22..omp_par, ptr %2, i32 100, i32 0, i8 0)
  call void @__kmpc_target_deinit()
  br label %worker.exit

worker.exit:                                      ; preds = %entry
  ret void
}


define internal void @__omp_offloading_10303_1849aab__QQmain_l22..omp_par(i32 %0, ptr %1) {
omp_loop.body:
  %gep = getelementptr { ptr, ptr }, ptr %1, i32 0, i32 1
  %p = load ptr, ptr %gep, align 8
  %5 = add i32 %0, 1
  store i32 %5, ptr %p, align 4
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr addrspacecast (ptr addrspace(1) @1 to ptr))
  call void @__kmpc_parallel_60(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i32 %omp_global_thread_num, i32 1, i32 -1, i32 -1, ptr @parallel_func_..omp_par.3, ptr @parallel_func_..omp_par.3.wrapper, ptr %1, i64 1, i32 0)
  %6 = load i32, ptr %p, align 4
  %7 = add i32 %6, 1
  store i32 %7, ptr %p, align 4
  ret void
}

define internal void @parallel_func_..omp_par.3.wrapper(i16 noundef zeroext %0, i32 noundef %1) {
entry:
  %addr = alloca i32, align 4, addrspace(5)
  %addr.ascast = addrspacecast ptr addrspace(5) %addr to ptr
  %zero = alloca i32, align 4, addrspace(5)
  %zero.ascast = addrspacecast ptr addrspace(5) %zero to ptr
  %global_args = alloca ptr, align 8, addrspace(5)
  %global_args.ascast = addrspacecast ptr addrspace(5) %global_args to ptr
  store i32 %1, ptr %addr.ascast, align 4
  store i32 0, ptr %zero.ascast, align 4
  call void @__kmpc_get_shared_variables(ptr %global_args.ascast)
  %2 = load ptr, ptr %global_args.ascast, align 8
  %3 = getelementptr inbounds ptr, ptr %2, i64 0
  %structArg = load ptr, ptr %3, align 8
  call void @parallel_func_..omp_par.3(ptr %addr.ascast, ptr %zero.ascast, ptr %structArg)
  ret void
}


declare void @__kmpc_get_shared_variables(ptr)
declare i32 @__kmpc_target_init(ptr, ptr)
declare noalias ptr @__kmpc_alloc_shared(i64)
declare void @__kmpc_target_deinit()
declare i32 @__kmpc_global_thread_num(ptr)
declare void @__kmpc_parallel_60(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64, i32)
declare void @__kmpc_distribute_static_loop_4u(ptr, ptr, ptr, i32, i32, i8)

attributes #1 = { nounwind "frame-pointer"="all" }
attributes #4 = { "kernel" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp-device", i32 52}
!1 = !{i32 7, !"openmp", i32 52}

; CHECK: @__omp_offloading_{{.*}}_kernel_environment = {{.*}}%struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 {{[0-9]+}}, i8 {{[0-9]+}}, i8 3,
; CHECK: define internal void @__omp_offloading_10303_1849aab__QQmain_l22..omp_par(
; CHECK: region.guarded:
; CHECK: region.guarded{{[0-9]+}}:
; CHECK: ret void
