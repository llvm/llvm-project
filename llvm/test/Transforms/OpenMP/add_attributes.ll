; RUN: opt < %s -S -passes=openmp-opt-cgscc -mtriple=x86_64-unknown-unknown | FileCheck %s
; RUN: opt < %s -S -passes=openmp-opt-cgscc -openmp-ir-builder-optimistic-attributes -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=OPTIMISTIC
; RUN: opt < %s -S -passes=openmp-opt-cgscc -mtriple=s390x-unknown-linux | FileCheck %s --check-prefix=EXT
; RUN: opt < %s -S -passes=openmp-opt-cgscc -mtriple=mips-linux-gnu | FileCheck %s --check-prefix=MIPS_EXT
; RUN: opt < %s -S -passes=openmp-opt-cgscc -mtriple=riscv64 | FileCheck %s --check-prefix=RISCV_EXT
; REQUIRES: x86-registered-target, systemz-registered-target, mips-registered-target, riscv-registered-target

%struct.omp_lock_t = type { ptr }
%struct.omp_nest_lock_t = type { ptr }
%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.__tgt_async_info = type { ptr }
%struct.__tgt_kernel_arguments = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, i64 }

define void @call_all(i32 %schedule, ptr %lock, i32 %lock_hint, ptr %nest_lock, i32 %i, ptr %s, i64 %st, ptr %vp, double %d, i32 %proc_bind, i64 %allocator_handle, ptr %cp, i64 %event_handle, i32 %pause_resource) {
entry:
  %schedule.addr = alloca i32, align 4
  %lock.addr = alloca ptr, align 8
  %lock_hint.addr = alloca i32, align 4
  %nest_lock.addr = alloca ptr, align 8
  %i.addr = alloca i32, align 4
  %s.addr = alloca ptr, align 8
  %st.addr = alloca i64, align 8
  %vp.addr = alloca ptr, align 8
  %d.addr = alloca double, align 8
  %proc_bind.addr = alloca i32, align 4
  %allocator_handle.addr = alloca i64, align 8
  %cp.addr = alloca ptr, align 8
  %event_handle.addr = alloca i64, align 8
  %pause_resource.addr = alloca i32, align 4
  store i32 %schedule, ptr %schedule.addr, align 4
  store ptr %lock, ptr %lock.addr, align 8
  store i32 %lock_hint, ptr %lock_hint.addr, align 4
  store ptr %nest_lock, ptr %nest_lock.addr, align 8
  store i32 %i, ptr %i.addr, align 4
  store ptr %s, ptr %s.addr, align 8
  store i64 %st, ptr %st.addr, align 8
  store ptr %vp, ptr %vp.addr, align 8
  store double %d, ptr %d.addr, align 8
  store i32 %proc_bind, ptr %proc_bind.addr, align 4
  store i64 %allocator_handle, ptr %allocator_handle.addr, align 8
  store ptr %cp, ptr %cp.addr, align 8
  store i64 %event_handle, ptr %event_handle.addr, align 8
  store i32 %pause_resource, ptr %pause_resource.addr, align 4
  call void @omp_set_num_threads(i32 0)
  call void @omp_set_dynamic(i32 0)
  call void @omp_set_nested(i32 0)
  call void @omp_set_max_active_levels(i32 0)
  %0 = load i32, ptr %schedule.addr, align 4
  call void @omp_set_schedule(i32 %0, i32 0)
  %call = call i32 @omp_get_num_threads()
  store i32 %call, ptr %i.addr, align 4
  %1 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %1)
  %call1 = call i32 @omp_get_dynamic()
  store i32 %call1, ptr %i.addr, align 4
  %2 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %2)
  %call2 = call i32 @omp_get_nested()
  store i32 %call2, ptr %i.addr, align 4
  %3 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %3)
  %call3 = call i32 @omp_get_max_threads()
  store i32 %call3, ptr %i.addr, align 4
  %4 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %4)
  %call4 = call i32 @omp_get_thread_num()
  store i32 %call4, ptr %i.addr, align 4
  %5 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %5)
  %call5 = call i32 @omp_get_num_procs()
  store i32 %call5, ptr %i.addr, align 4
  %6 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %6)
  %call6 = call i32 @omp_in_parallel()
  store i32 %call6, ptr %i.addr, align 4
  %7 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %7)
  %call7 = call i32 @omp_in_final()
  store i32 %call7, ptr %i.addr, align 4
  %8 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %8)
  %call8 = call i32 @omp_get_active_level()
  store i32 %call8, ptr %i.addr, align 4
  %9 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %9)
  %call9 = call i32 @omp_get_level()
  store i32 %call9, ptr %i.addr, align 4
  %10 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %10)
  %call10 = call i32 @omp_get_ancestor_thread_num(i32 0)
  store i32 %call10, ptr %i.addr, align 4
  %11 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %11)
  %call11 = call i32 @omp_get_team_size(i32 0)
  store i32 %call11, ptr %i.addr, align 4
  %12 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %12)
  %call12 = call i32 @omp_get_thread_limit()
  store i32 %call12, ptr %i.addr, align 4
  %13 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %13)
  %call13 = call i32 @omp_get_max_active_levels()
  store i32 %call13, ptr %i.addr, align 4
  %14 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %14)
  call void @omp_get_schedule(ptr %schedule.addr, ptr %i.addr)
  %call14 = call i32 @omp_get_max_task_priority()
  store i32 %call14, ptr %i.addr, align 4
  %15 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %15)
  %16 = load ptr, ptr %lock.addr, align 8
  call void @omp_init_lock(ptr %16)
  %17 = load ptr, ptr %lock.addr, align 8
  call void @omp_set_lock(ptr %17)
  %18 = load ptr, ptr %lock.addr, align 8
  call void @omp_unset_lock(ptr %18)
  %19 = load ptr, ptr %lock.addr, align 8
  call void @omp_destroy_lock(ptr %19)
  %20 = load ptr, ptr %lock.addr, align 8
  %call15 = call i32 @omp_test_lock(ptr %20)
  store i32 %call15, ptr %i.addr, align 4
  %21 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %21)
  %22 = load ptr, ptr %nest_lock.addr, align 8
  call void @omp_init_nest_lock(ptr %22)
  %23 = load ptr, ptr %nest_lock.addr, align 8
  call void @omp_set_nest_lock(ptr %23)
  %24 = load ptr, ptr %nest_lock.addr, align 8
  call void @omp_unset_nest_lock(ptr %24)
  %25 = load ptr, ptr %nest_lock.addr, align 8
  call void @omp_destroy_nest_lock(ptr %25)
  %26 = load ptr, ptr %nest_lock.addr, align 8
  %call16 = call i32 @omp_test_nest_lock(ptr %26)
  store i32 %call16, ptr %i.addr, align 4
  %27 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %27)
  %28 = load ptr, ptr %lock.addr, align 8
  %29 = load i32, ptr %lock_hint.addr, align 4
  call void @omp_init_lock_with_hint(ptr %28, i32 %29)
  %30 = load ptr, ptr %nest_lock.addr, align 8
  %31 = load i32, ptr %lock_hint.addr, align 4
  call void @omp_init_nest_lock_with_hint(ptr %30, i32 %31)
  %call17 = call double @omp_get_wtime()
  store double %call17, ptr %d.addr, align 8
  %32 = load double, ptr %d.addr, align 8
  call void @use_double(double %32)
  %call18 = call double @omp_get_wtick()
  store double %call18, ptr %d.addr, align 8
  %33 = load double, ptr %d.addr, align 8
  call void @use_double(double %33)
  %call19 = call i32 @omp_get_default_device()
  store i32 %call19, ptr %i.addr, align 4
  %34 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %34)
  call void @omp_set_default_device(i32 0)
  %call20 = call i32 @omp_is_initial_device()
  store i32 %call20, ptr %i.addr, align 4
  %35 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %35)
  %call21 = call i32 @omp_get_num_devices()
  store i32 %call21, ptr %i.addr, align 4
  %36 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %36)
  %call22 = call i32 @omp_get_num_teams()
  store i32 %call22, ptr %i.addr, align 4
  %37 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %37)
  %call23 = call i32 @omp_get_team_num()
  store i32 %call23, ptr %i.addr, align 4
  %38 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %38)
  %call24 = call i32 @omp_get_cancellation()
  store i32 %call24, ptr %i.addr, align 4
  %39 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %39)
  %call25 = call i32 @omp_get_initial_device()
  store i32 %call25, ptr %i.addr, align 4
  %40 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %40)
  %41 = load i64, ptr %st.addr, align 8
  %42 = load i32, ptr %i.addr, align 4
  %call26 = call ptr @omp_target_alloc(i64 %41, i32 %42)
  store ptr %call26, ptr %vp.addr, align 8
  %43 = load ptr, ptr %vp.addr, align 8
  call void @use_voidptr(ptr %43)
  %44 = load ptr, ptr %vp.addr, align 8
  %45 = load i32, ptr %i.addr, align 4
  call void @omp_target_free(ptr %44, i32 %45)
  %46 = load ptr, ptr %vp.addr, align 8
  %47 = load i32, ptr %i.addr, align 4
  %call27 = call i32 @omp_target_is_present(ptr %46, i32 %47)
  store i32 %call27, ptr %i.addr, align 4
  %48 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %48)
  %49 = load ptr, ptr %vp.addr, align 8
  %50 = load ptr, ptr %vp.addr, align 8
  %51 = load i64, ptr %st.addr, align 8
  %52 = load i64, ptr %st.addr, align 8
  %53 = load i64, ptr %st.addr, align 8
  %54 = load i32, ptr %i.addr, align 4
  %55 = load i32, ptr %i.addr, align 4
  %call28 = call i32 @omp_target_memcpy(ptr %49, ptr %50, i64 %51, i64 %52, i64 %53, i32 %54, i32 %55)
  store i32 %call28, ptr %i.addr, align 4
  %56 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %56)
  %57 = load ptr, ptr %vp.addr, align 8
  %58 = load ptr, ptr %vp.addr, align 8
  %59 = load i64, ptr %st.addr, align 8
  %60 = load i64, ptr %st.addr, align 8
  %61 = load i32, ptr %i.addr, align 4
  %call29 = call i32 @omp_target_associate_ptr(ptr %57, ptr %58, i64 %59, i64 %60, i32 %61)
  store i32 %call29, ptr %i.addr, align 4
  %62 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %62)
  %63 = load ptr, ptr %vp.addr, align 8
  %64 = load i32, ptr %i.addr, align 4
  %call30 = call i32 @omp_target_disassociate_ptr(ptr %63, i32 %64)
  store i32 %call30, ptr %i.addr, align 4
  %65 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %65)
  %call31 = call i32 @omp_get_device_num()
  store i32 %call31, ptr %i.addr, align 4
  %66 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %66)
  %call32 = call i32 @omp_get_proc_bind()
  store i32 %call32, ptr %proc_bind.addr, align 4
  %call33 = call i32 @omp_get_num_places()
  store i32 %call33, ptr %i.addr, align 4
  %67 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %67)
  %call34 = call i32 @omp_get_place_num_procs(i32 0)
  store i32 %call34, ptr %i.addr, align 4
  %68 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %68)
  %69 = load i32, ptr %i.addr, align 4
  call void @omp_get_place_proc_ids(i32 %69, ptr %i.addr)
  %call35 = call i32 @omp_get_place_num()
  store i32 %call35, ptr %i.addr, align 4
  %70 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %70)
  %call36 = call i32 @omp_get_partition_num_places()
  store i32 %call36, ptr %i.addr, align 4
  %71 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %71)
  call void @omp_get_partition_place_nums(ptr %i.addr)
  %72 = load i32, ptr %i.addr, align 4
  %73 = load i32, ptr %i.addr, align 4
  %74 = load ptr, ptr %vp.addr, align 8
  %call37 = call i32 @omp_control_tool(i32 %72, i32 %73, ptr %74)
  store i32 %call37, ptr %i.addr, align 4
  %75 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %75)
  %76 = load i64, ptr %allocator_handle.addr, align 8
  call void @omp_destroy_allocator(i64 %76)
  %77 = load i64, ptr %allocator_handle.addr, align 8
  call void @omp_set_default_allocator(i64 %77)
  %call38 = call i64 @omp_get_default_allocator()
  store i64 %call38, ptr %allocator_handle.addr, align 8
  %78 = load i64, ptr %st.addr, align 8
  %79 = load i64, ptr %allocator_handle.addr, align 8
  %call39 = call ptr @omp_alloc(i64 %78, i64 %79)
  store ptr %call39, ptr %vp.addr, align 8
  %80 = load ptr, ptr %vp.addr, align 8
  call void @use_voidptr(ptr %80)
  %81 = load ptr, ptr %vp.addr, align 8
  %82 = load i64, ptr %allocator_handle.addr, align 8
  call void @omp_free(ptr %81, i64 %82)
  %83 = load i64, ptr %st.addr, align 8
  %84 = load i64, ptr %allocator_handle.addr, align 8
  %call40 = call ptr @omp_alloc(i64 %83, i64 %84)
  store ptr %call40, ptr %vp.addr, align 8
  %85 = load ptr, ptr %vp.addr, align 8
  call void @use_voidptr(ptr %85)
  %86 = load ptr, ptr %vp.addr, align 8
  %87 = load i64, ptr %allocator_handle.addr, align 8
  call void @omp_free(ptr %86, i64 %87)
  %88 = load ptr, ptr %s.addr, align 8
  call void @ompc_set_affinity_format(ptr %88)
  %89 = load ptr, ptr %cp.addr, align 8
  %90 = load i64, ptr %st.addr, align 8
  %call41 = call i64 @ompc_get_affinity_format(ptr %89, i64 %90)
  store i64 %call41, ptr %st.addr, align 8
  %91 = load i64, ptr %st.addr, align 8
  call void @use_sizet(i64 %91)
  %92 = load ptr, ptr %s.addr, align 8
  call void @ompc_display_affinity(ptr %92)
  %93 = load ptr, ptr %cp.addr, align 8
  %94 = load i64, ptr %st.addr, align 8
  %95 = load ptr, ptr %s.addr, align 8
  %call42 = call i64 @ompc_capture_affinity(ptr %93, i64 %94, ptr %95)
  store i64 %call42, ptr %st.addr, align 8
  %96 = load i64, ptr %st.addr, align 8
  call void @use_sizet(i64 %96)
  %97 = load i64, ptr %event_handle.addr, align 8
  call void @omp_fulfill_event(i64 %97)
  %98 = load i32, ptr %pause_resource.addr, align 4
  %99 = load i32, ptr %i.addr, align 4
  %call43 = call i32 @omp_pause_resource(i32 %98, i32 %99)
  store i32 %call43, ptr %i.addr, align 4
  %100 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %100)
  %101 = load i32, ptr %pause_resource.addr, align 4
  %call44 = call i32 @omp_pause_resource_all(i32 %101)
  store i32 %call44, ptr %i.addr, align 4
  %102 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %102)
  %call45 = call i32 @omp_get_supported_active_levels()
  store i32 %call45, ptr %i.addr, align 4
  %103 = load i32, ptr %i.addr, align 4
  call void @use_int(i32 %103)
  ret void
}

declare dso_local void @omp_set_num_threads(i32)

declare dso_local void @omp_set_dynamic(i32)

declare dso_local void @omp_set_nested(i32)

declare dso_local void @omp_set_max_active_levels(i32)

declare dso_local void @omp_set_schedule(i32, i32)

declare dso_local i32 @omp_get_num_threads()

declare dso_local void @use_int(i32)

declare dso_local i32 @omp_get_dynamic()

declare dso_local i32 @omp_get_nested()

declare dso_local i32 @omp_get_max_threads()

declare dso_local i32 @omp_get_thread_num()

declare dso_local i32 @omp_get_num_procs()

declare dso_local i32 @omp_in_parallel()

declare dso_local i32 @omp_in_final()

declare dso_local i32 @omp_get_active_level()

declare dso_local i32 @omp_get_level()

declare dso_local i32 @omp_get_ancestor_thread_num(i32)

declare dso_local i32 @omp_get_team_size(i32)

declare dso_local i32 @omp_get_thread_limit()

declare dso_local i32 @omp_get_max_active_levels()

declare dso_local void @omp_get_schedule(ptr, ptr)

declare dso_local i32 @omp_get_max_task_priority()

declare dso_local void @omp_init_lock(ptr)

declare dso_local void @omp_set_lock(ptr)

declare dso_local void @omp_unset_lock(ptr)

declare dso_local void @omp_destroy_lock(ptr)

declare dso_local i32 @omp_test_lock(ptr)

declare dso_local void @omp_init_nest_lock(ptr)

declare dso_local void @omp_set_nest_lock(ptr)

declare dso_local void @omp_unset_nest_lock(ptr)

declare dso_local void @omp_destroy_nest_lock(ptr)

declare dso_local i32 @omp_test_nest_lock(ptr)

declare dso_local void @omp_init_lock_with_hint(ptr, i32)

declare dso_local void @omp_init_nest_lock_with_hint(ptr, i32)

declare dso_local double @omp_get_wtime()

declare dso_local void @use_double(double)

declare dso_local double @omp_get_wtick()

declare dso_local i32 @omp_get_default_device()

declare dso_local void @omp_set_default_device(i32)

declare dso_local i32 @omp_is_initial_device()

declare dso_local i32 @omp_get_num_devices()

declare dso_local i32 @omp_get_num_teams()

declare dso_local i32 @omp_get_team_num()

declare dso_local i32 @omp_get_cancellation()

declare dso_local i32 @omp_get_initial_device()

declare dso_local ptr @omp_target_alloc(i64, i32)

declare dso_local void @use_voidptr(ptr)

declare dso_local void @omp_target_free(ptr, i32)

declare dso_local i32 @omp_target_is_present(ptr, i32)

declare dso_local i32 @omp_target_memcpy(ptr, ptr, i64, i64, i64, i32, i32)

declare dso_local i32 @omp_target_associate_ptr(ptr, ptr, i64, i64, i32)

declare dso_local i32 @omp_target_disassociate_ptr(ptr, i32)

declare dso_local i32 @omp_get_device_num()

declare dso_local i32 @omp_get_proc_bind()

declare dso_local i32 @omp_get_num_places()

declare dso_local i32 @omp_get_place_num_procs(i32)

declare dso_local void @omp_get_place_proc_ids(i32, ptr)

declare dso_local i32 @omp_get_place_num()

declare dso_local i32 @omp_get_partition_num_places()

declare dso_local void @omp_get_partition_place_nums(ptr)

declare dso_local i32 @omp_control_tool(i32, i32, ptr)

declare dso_local void @omp_destroy_allocator(i64)

declare dso_local void @omp_set_default_allocator(i64)

declare dso_local i64 @omp_get_default_allocator()

declare dso_local ptr @omp_alloc(i64, i64)

declare dso_local void @omp_free(ptr, i64)

declare dso_local void @ompc_set_affinity_format(ptr)

declare dso_local i64 @ompc_get_affinity_format(ptr, i64)

declare dso_local void @use_sizet(i64)

declare dso_local void @ompc_display_affinity(ptr)

declare dso_local i64 @ompc_capture_affinity(ptr, i64, ptr)

declare dso_local void @omp_fulfill_event(i64)

declare dso_local i32 @omp_pause_resource(i32, i32)

declare dso_local i32 @omp_pause_resource_all(i32)

declare dso_local i32 @omp_get_supported_active_levels()

declare void @__kmpc_barrier(ptr, i32)

declare i32 @__kmpc_cancel(ptr, i32, i32)

declare i32 @__kmpc_cancel_barrier(ptr, i32)

declare void @__kmpc_flush(ptr)

declare i32 @__kmpc_global_thread_num(ptr)

declare void @__kmpc_fork_call(ptr, i32, ptr, ...)

declare i32 @__kmpc_omp_taskwait(ptr, i32)

declare i32 @__kmpc_omp_taskyield(ptr, i32, i32)

declare void @__kmpc_push_num_threads(ptr, i32, i32)

declare void @__kmpc_push_proc_bind(ptr, i32, i32)

declare void @__kmpc_serialized_parallel(ptr, i32)

declare void @__kmpc_end_serialized_parallel(ptr, i32)

declare i32 @__kmpc_master(ptr, i32)

declare void @__kmpc_end_master(ptr, i32)

declare void @__kmpc_critical(ptr, i32, ptr)

declare void @__kmpc_critical_with_hint(ptr, i32, ptr, i32)

declare void @__kmpc_end_critical(ptr, i32, ptr)

declare void @__kmpc_begin(ptr, i32)

declare void @__kmpc_end(ptr)

declare i32 @__kmpc_reduce(ptr, i32, i32, i64, ptr, ptr, ptr)

declare i32 @__kmpc_reduce_nowait(ptr, i32, i32, i64, ptr, ptr, ptr)

declare void @__kmpc_end_reduce(ptr, i32, ptr)

declare void @__kmpc_end_reduce_nowait(ptr, i32, ptr)

declare void @__kmpc_ordered(ptr, i32)

declare void @__kmpc_end_ordered(ptr, i32)

declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_for_static_init_8(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)

declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)

declare void @__kmpc_for_static_fini(ptr, i32)

declare void @__kmpc_team_static_init_4(ptr, i32, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_team_static_init_4u(ptr, i32, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_team_static_init_8(ptr, i32, ptr, ptr, ptr, ptr, i64, i64)

declare void @__kmpc_team_static_init_8u(ptr, i32, ptr, ptr, ptr, ptr, i64, i64)

declare void @__kmpc_dist_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_dist_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_dist_for_static_init_8(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i64, i64)

declare void @__kmpc_dist_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i64, i64)

declare i32 @__kmpc_single(ptr, i32)

declare void @__kmpc_end_single(ptr, i32)

declare ptr @__kmpc_omp_task_alloc(ptr, i32, i32, i64, i64, ptr)

declare i32 @__kmpc_omp_task(ptr, i32, ptr)

declare void @__kmpc_end_taskgroup(ptr, i32)

declare void @__kmpc_taskgroup(ptr, i32)

declare void @__kmpc_dist_dispatch_init_4(ptr, i32, i32, ptr, i32, i32, i32, i32)

declare void @__kmpc_dist_dispatch_init_4u(ptr, i32, i32, ptr, i32, i32, i32, i32)

declare void @__kmpc_dist_dispatch_init_8(ptr, i32, i32, ptr, i64, i64, i64, i64)

declare void @__kmpc_dist_dispatch_init_8u(ptr, i32, i32, ptr, i64, i64, i64, i64)

declare void @__kmpc_dispatch_init_4(ptr, i32, i32, i32, i32, i32, i32)

declare void @__kmpc_dispatch_init_4u(ptr, i32, i32, i32, i32, i32, i32)

declare void @__kmpc_dispatch_init_8(ptr, i32, i32, i64, i64, i64, i64)

declare void @__kmpc_dispatch_init_8u(ptr, i32, i32, i64, i64, i64, i64)

declare i32 @__kmpc_dispatch_next_4(ptr, i32, ptr, ptr, ptr, ptr)

declare i32 @__kmpc_dispatch_next_4u(ptr, i32, ptr, ptr, ptr, ptr)

declare i32 @__kmpc_dispatch_next_8(ptr, i32, ptr, ptr, ptr, ptr)

declare i32 @__kmpc_dispatch_next_8u(ptr, i32, ptr, ptr, ptr, ptr)

declare void @__kmpc_dispatch_fini_4(ptr, i32)

declare void @__kmpc_dispatch_fini_4u(ptr, i32)

declare void @__kmpc_dispatch_fini_8(ptr, i32)

declare void @__kmpc_dispatch_fini_8u(ptr, i32)

declare void @__kmpc_omp_task_begin_if0(ptr, i32, ptr)

declare void @__kmpc_omp_task_complete_if0(ptr, i32, ptr)

declare i32 @__kmpc_omp_task_with_deps(ptr, i32, ptr, i32, ptr, i32, ptr)

declare void @__kmpc_omp_wait_deps(ptr, i32, i32, ptr, i32, ptr)

declare i32 @__kmpc_cancellationpoint(ptr, i32, i32)

declare void @__kmpc_push_num_teams(ptr, i32, i32, i32)

declare void @__kmpc_fork_teams(ptr, i32, ptr, ...)

declare void @__kmpc_taskloop(ptr, i32, ptr, i32, ptr, ptr, i64, i32, i32, i64, ptr)

declare ptr @__kmpc_omp_target_task_alloc(ptr, i32, i32, i64, i64, ptr, i64)

declare ptr @__kmpc_taskred_modifier_init(ptr, i32, i32, i32, ptr)

declare ptr @__kmpc_taskred_init(i32, i32, ptr)

declare void @__kmpc_task_reduction_modifier_fini(ptr, i32, i32)

declare void @__kmpc_copyprivate(ptr, i32, i64, ptr, ptr, i32)

declare ptr @__kmpc_threadprivate_cached(ptr, i32, ptr, i64, ptr)

declare void @__kmpc_threadprivate_register(ptr, ptr, ptr, ptr, ptr)

declare void @__kmpc_doacross_init(ptr, i32, i32, ptr)

declare void @__kmpc_doacross_wait(ptr, i32, ptr)

declare void @__kmpc_doacross_post(ptr, i32, ptr)

declare void @__kmpc_doacross_fini(ptr, i32)

declare ptr @__kmpc_alloc(i32, i64, ptr)

declare void @__kmpc_free(i32, ptr, ptr)

declare ptr @__kmpc_init_allocator(i32, ptr, i32, ptr)

declare void @__kmpc_destroy_allocator(i32, ptr)

declare void @__kmpc_push_target_tripcount_mapper(ptr, i64, i64)

declare i64 @__kmpc_warp_active_thread_mask()

declare void @__kmpc_syncwarp(i64)

declare i32 @__tgt_target_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare i32 @__tgt_target_nowait_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, ptr)

declare i32 @__tgt_target_teams_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32)

declare i32 @__tgt_target_teams_nowait_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, ptr, i32, ptr)

declare void @__tgt_register_requires(i64)

declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare void @__tgt_target_data_begin_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare void @__tgt_target_data_end_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare void @__tgt_target_data_update_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare void @__tgt_target_data_update_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

declare i64 @__tgt_mapper_num_components(ptr)

declare void @__tgt_push_mapper_component(ptr, ptr, ptr, i64, i64, ptr)

declare ptr @__kmpc_task_allow_completion_event(ptr, i32, ptr)

declare ptr @__kmpc_task_reduction_get_th_data(i32, ptr, ptr)

declare ptr @__kmpc_task_reduction_init(i32, i32, ptr)

declare ptr @__kmpc_task_reduction_modifier_init(ptr, i32, i32, i32, ptr)

declare void @__kmpc_proxy_task_completed_ooo(ptr)

; Function Attrs: noinline cold
declare void @__kmpc_barrier_simple_spmd(ptr nocapture nofree readonly, i32) #0

attributes #0 = { noinline cold }

declare ptr @__kmpc_aligned_alloc(i32, i64, i64, ptr);

declare ptr @__kmpc_alloc_shared(i64);

declare void @__kmpc_barrier_simple_generic(ptr, i32);

declare void @__kmpc_begin_sharing_variables(ptr, i64);

declare void @__kmpc_distribute_static_fini(ptr, i32);

declare void @__kmpc_distribute_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32);

declare void @__kmpc_distribute_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32);

declare void @__kmpc_distribute_static_init_8(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64);

declare void @__kmpc_distribute_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64);

declare void @__kmpc_end_masked(ptr, i32);

declare void @__kmpc_end_sharing_variables();

declare void @__kmpc_error(ptr, i32, ptr);

declare void @__kmpc_fork_call_if(ptr, i32, ptr, i32, ptr);

declare void @__kmpc_free_shared(ptr, i64);

declare i32 @__kmpc_get_hardware_num_blocks();

declare i32 @__kmpc_get_hardware_num_threads_in_block();

declare i32 @__kmpc_get_hardware_thread_id_in_block();

declare void @__kmpc_get_shared_variables(ptr);

declare i32 @__kmpc_get_warp_size();

declare i8 @__kmpc_is_spmd_exec_mode();

declare void @__kmpc_kernel_end_parallel();

declare i1 @__kmpc_kernel_parallel(ptr);

declare void @__kmpc_kernel_prepare_parallel(ptr);

declare i32 @__kmpc_masked(ptr, i32, i32);

declare void @__kmpc_nvptx_end_reduce_nowait(i32);

declare i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(ptr, i32, i32, i64, ptr, ptr, ptr);

declare i32 @__kmpc_nvptx_teams_reduce_nowait_v2(ptr, i32, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr);

declare i32 @__kmpc_omp_reg_task_with_affinity(ptr, i32, ptr, i32, ptr);

declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64);

declare i32 @__kmpc_shuffle_int32(i32, i16, i16);

declare i64 @__kmpc_shuffle_int64(i64, i16, i16);

declare void @__kmpc_target_deinit(ptr, i8);

declare i32 @__kmpc_target_init(ptr, i8, i1);

declare void @__tgt_interop_destroy(ptr, i32, ptr, i32, i32, ptr, i32);

declare void @__tgt_interop_init(ptr, i32, ptr, i32, i32, i32, ptr, i32);

declare void @__tgt_interop_use(ptr, i32, ptr, i32, i32, ptr, i32);

declare void @__tgt_target_data_begin_mapper_issue(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr);

declare void @__tgt_target_data_begin_mapper_wait(i64, ptr);

declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr);

declare i32 @__tgt_target_kernel_nowait(ptr, i64, i32, i32, ptr, ptr, i32, ptr, i32, ptr);


; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_set_num_threads(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_set_dynamic(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_set_nested(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_set_max_active_levels(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_set_schedule(i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_num_threads()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @use_int(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_dynamic()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_nested()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_max_threads()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_thread_num()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_num_procs()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_in_parallel()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_in_final()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_active_level()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_level()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_ancestor_thread_num(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_team_size(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_thread_limit()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_max_active_levels()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_get_schedule(ptr nocapture writeonly, ptr nocapture writeonly)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_max_task_priority()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_set_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_unset_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_destroy_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_test_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_nest_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_set_nest_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_unset_nest_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_destroy_nest_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_test_nest_lock(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_lock_with_hint(ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_nest_lock_with_hint(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local double @omp_get_wtime()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @use_double(double)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local double @omp_get_wtick()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_default_device()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_set_default_device(i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_is_initial_device()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_num_devices()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_num_teams()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_team_num()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_cancellation()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_initial_device()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local ptr @omp_target_alloc(i64, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @use_voidptr(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_target_free(ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_is_present(ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_memcpy(ptr, ptr, i64, i64, i64, i32, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_associate_ptr(ptr, ptr, i64, i64, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_disassociate_ptr(ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_device_num()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_proc_bind()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_num_places()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_place_num_procs(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_get_place_proc_ids(i32, ptr nocapture writeonly)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_place_num()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_partition_num_places()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_get_partition_place_nums(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_control_tool(i32, i32, ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_destroy_allocator(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_set_default_allocator(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i64 @omp_get_default_allocator()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local ptr @omp_alloc(i64, i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_free(ptr, i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @ompc_set_affinity_format(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i64 @ompc_get_affinity_format(ptr, i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @use_sizet(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @ompc_display_affinity(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i64 @ompc_capture_affinity(ptr, i64, ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_fulfill_event(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_pause_resource(i32, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_pause_resource_all(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_supported_active_levels()

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_barrier(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_cancel(ptr, i32, i32)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare i32 @__kmpc_cancel_barrier(ptr, i32)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_flush(ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_global_thread_num(ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_fork_call(ptr, i32, ptr, ...)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_taskwait(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_taskyield(ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_num_threads(ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_proc_bind(ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_serialized_parallel(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_serialized_parallel(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_master(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_master(ptr, i32)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_critical(ptr, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_critical_with_hint(ptr, i32, ptr, i32)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_end_critical(ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_begin(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end(ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare i32 @__kmpc_reduce(ptr, i32, i32, i64, ptr, ptr, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare i32 @__kmpc_reduce_nowait(ptr, i32, i32, i64, ptr, ptr, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_end_reduce(ptr, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_end_reduce_nowait(ptr, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_ordered(ptr, i32)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_end_ordered(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_8(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_fini(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_4(ptr, i32, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_4u(ptr, i32, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_8(ptr, i32, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_8u(ptr, i32, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_8(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare i32 @__kmpc_single(ptr, i32)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_end_single(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_omp_task_alloc(ptr, i32, i32, i64, i64, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_task(ptr, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_end_taskgroup(ptr, i32)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_taskgroup(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_4(ptr, i32, i32, ptr, i32, i32, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_4u(ptr, i32, i32, ptr, i32, i32, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_8(ptr, i32, i32, ptr, i64, i64, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_8u(ptr, i32, i32, ptr, i64, i64, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_4(ptr, i32, i32, i32, i32, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_4u(ptr, i32, i32, i32, i32, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_8(ptr, i32, i32, i64, i64, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_8u(ptr, i32, i32, i64, i64, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_4(ptr, i32, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_4u(ptr, i32, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_8(ptr, i32, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_8u(ptr, i32, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_4(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_4u(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_8(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_8u(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_omp_task_begin_if0(ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_omp_task_complete_if0(ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_task_with_deps(ptr, i32, ptr, i32, ptr, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_omp_wait_deps(ptr, i32, i32, ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_cancellationpoint(ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_num_teams(ptr, i32, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_fork_teams(ptr, i32, ptr, ...)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_taskloop(ptr, i32, ptr, i32, ptr, ptr, i64, i32, i32, i64, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_omp_target_task_alloc(ptr, i32, i32, i64, i64, ptr, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_taskred_modifier_init(ptr, i32, i32, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_taskred_init(i32, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_task_reduction_modifier_fini(ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_copyprivate(ptr, i32, i64, ptr, ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_threadprivate_cached(ptr, i32, ptr, i64, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_threadprivate_register(ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_init(ptr, i32, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_wait(ptr, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_post(ptr, i32, ptr)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_fini(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_alloc(i32, i64, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_free(i32, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_init_allocator(i32, ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_destroy_allocator(i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_target_tripcount_mapper(ptr, i64, i64)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare i64 @__kmpc_warp_active_thread_mask()

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_syncwarp(i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_nowait_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_teams_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_teams_nowait_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_register_requires(i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_begin_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_end_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_update_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_update_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i64 @__tgt_mapper_num_components(ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_push_mapper_component(ptr, ptr, ptr, i64, i64, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_task_allow_completion_event(ptr, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_task_reduction_get_th_data(i32, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_task_reduction_init(i32, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_task_reduction_modifier_init(ptr, i32, i32, i32, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_proxy_task_completed_ooo(ptr)

; CHECK: ; Function Attrs: cold convergent noinline nounwind
; CHECK-NEXT: declare void @__kmpc_barrier_simple_spmd(ptr nocapture nofree readonly, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare ptr @__kmpc_aligned_alloc(i32, i64, i64, ptr)

; CHECK: ; Function Attrs: nosync nounwind allocsize(0)
; CHECK-NEXT: declare ptr @__kmpc_alloc_shared(i64)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK: declare void @__kmpc_barrier_simple_generic(ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_begin_sharing_variables(ptr, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare void @__kmpc_distribute_static_fini(ptr, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare void @__kmpc_distribute_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare void @__kmpc_distribute_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare void @__kmpc_distribute_static_init_8(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare void @__kmpc_distribute_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare void @__kmpc_end_masked(ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_end_sharing_variables()

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_error(ptr, i32, ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_fork_call_if(ptr, i32, ptr, i32, ptr)

; CHECK: ; Function Attrs: nosync nounwind
; CHECK-NEXT: declare void @__kmpc_free_shared(ptr allocptr nocapture, i64)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_get_hardware_num_blocks()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_get_hardware_num_threads_in_block()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_get_hardware_thread_id_in_block()

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_get_shared_variables(ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_get_warp_size()

; CHECK-NOT: Function Attrs
; CHECK: declare i8 @__kmpc_is_spmd_exec_mode()

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_kernel_end_parallel()

; CHECK-NOT: Function Attrs
; CHECK: declare i1 @__kmpc_kernel_parallel(ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_kernel_prepare_parallel(ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare i32 @__kmpc_masked(ptr, i32, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_nvptx_end_reduce_nowait(i32)

; CHECK-NOT: Function Attrs
; CHECK: declare i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(ptr, i32, i32, i64, ptr, ptr, ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare i32 @__kmpc_nvptx_teams_reduce_nowait_v2(ptr, i32, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare i32 @__kmpc_omp_reg_task_with_affinity(ptr, i32, ptr, i32, ptr)

; CHECK: ; Function Attrs: alwaysinline
; CHECK: declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64)

; CHECK-NOT: Function Attrs
; CHECK: declare i32 @__kmpc_shuffle_int32(i32, i16, i16)

; CHECK-NOT: Function Attrs
; CHECK: declare i64 @__kmpc_shuffle_int64(i64, i16, i16)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__kmpc_target_deinit(ptr, i8)

; CHECK-NOT: Function Attrs
; CHECK: declare i32 @__kmpc_target_init(ptr, i8, i1)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__tgt_interop_destroy(ptr, i32, ptr, i32, i32, ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__tgt_interop_init(ptr, i32, ptr, i32, i32, i32, ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__tgt_interop_use(ptr, i32, ptr, i32, i32, ptr, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__tgt_target_data_begin_mapper_issue(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

; CHECK-NOT: Function Attrs
; CHECK: declare void @__tgt_target_data_begin_mapper_wait(i64, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr)

; CHECK: ; Function Attrs: nounwind
; CHECK: declare i32 @__tgt_target_kernel_nowait(ptr, i64, i32, i32, ptr, ptr, i32, ptr, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: write)
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_num_threads(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: write)
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_dynamic(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: write)
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_nested(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: write)
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_max_active_levels(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: write)
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_schedule(i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_num_threads()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @use_int(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_dynamic()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_nested()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_max_threads()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_thread_num()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_num_procs()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_in_parallel()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_in_final()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_active_level()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_level()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_ancestor_thread_num(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_team_size(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_thread_limit()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_max_active_levels()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare dso_local void @omp_get_schedule(ptr nocapture writeonly, ptr nocapture writeonly)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_max_task_priority()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_set_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_unset_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_destroy_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_test_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_nest_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_set_nest_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_unset_nest_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_destroy_nest_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_test_nest_lock(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_lock_with_hint(ptr, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_nest_lock_with_hint(ptr, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare dso_local double @omp_get_wtime()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @use_double(double)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local double @omp_get_wtick()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_default_device()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_set_default_device(i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_is_initial_device()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_num_devices()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_num_teams()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_team_num()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_cancellation()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_initial_device()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local ptr @omp_target_alloc(i64, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @use_voidptr(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_target_free(ptr, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_is_present(ptr, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_memcpy(ptr, ptr, i64, i64, i64, i32, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_associate_ptr(ptr, ptr, i64, i64, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_disassociate_ptr(ptr, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_device_num()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_proc_bind()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_num_places()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_place_num_procs(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare dso_local void @omp_get_place_proc_ids(i32, ptr nocapture writeonly)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_place_num()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_partition_num_places()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local void @omp_get_partition_place_nums(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_control_tool(i32, i32, ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_destroy_allocator(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_set_default_allocator(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i64 @omp_get_default_allocator()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local ptr @omp_alloc(i64, i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_free(ptr, i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @ompc_set_affinity_format(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i64 @ompc_get_affinity_format(ptr, i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @use_sizet(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @ompc_display_affinity(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i64 @ompc_capture_affinity(ptr, i64, ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_fulfill_event(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_pause_resource(i32, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_pause_resource_all(i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_supported_active_levels()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_global_thread_num(ptr nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_fork_call(ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly, ...)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_taskwait(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_taskyield(ptr nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_push_num_threads(ptr nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_push_proc_bind(ptr nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_serialized_parallel(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_end_serialized_parallel(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_master(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_end_master(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_critical(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_critical_with_hint(ptr nocapture nofree readonly, i32, ptr, i32)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_critical(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_begin(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_end(ptr nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_reduce(ptr nocapture nofree readonly, i32, i32, i64, ptr nocapture nofree readonly, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_reduce_nowait(ptr nocapture nofree readonly, i32, i32, i64, ptr nocapture nofree readonly, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_reduce(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_reduce_nowait(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_ordered(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_ordered(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_4(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_4u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_8(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_8u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_fini(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_4(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_4u(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_8(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_8u(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_4(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_4u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_8(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_8u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_single(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_single(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_omp_task_alloc(ptr nocapture nofree readonly, i32, i32, i64, i64, ptr nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_task(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_taskgroup(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_taskgroup(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_4(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_4u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_8(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_8u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_4(ptr nocapture nofree readonly, i32, i32, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_4u(ptr nocapture nofree readonly, i32, i32, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_8(ptr nocapture nofree readonly, i32, i32, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_8u(ptr nocapture nofree readonly, i32, i32, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_4(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_4u(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_8(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_8u(ptr nocapture nofree readonly, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_4(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_4u(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_8(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_8u(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_omp_task_begin_if0(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_omp_task_complete_if0(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_task_with_deps(ptr nocapture nofree readonly, i32, ptr, i32, ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_omp_wait_deps(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_cancellationpoint(ptr nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC-NEXT: declare void @__kmpc_push_num_teams(ptr nocapture nofree readonly, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_fork_teams(ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly, ...)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_taskloop(ptr nocapture nofree readonly, i32, ptr, i32, ptr nocapture nofree, ptr nocapture nofree, i64, i32, i32, i64, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_omp_target_task_alloc(ptr nocapture nofree readonly, i32, i32, i64, i64, ptr nocapture nofree readonly, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_taskred_modifier_init(ptr nocapture nofree readonly, i32, i32, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare ptr @__kmpc_taskred_init(i32, i32, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_task_reduction_modifier_fini(ptr nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_copyprivate(ptr nocapture nofree readonly, i32, i64, ptr nocapture nofree readonly, ptr, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_threadprivate_cached(ptr nocapture nofree readonly, i32, ptr, i64, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_threadprivate_register(ptr nocapture nofree readonly, ptr, ptr nocapture nofree readonly, ptr nocapture nofree readonly, ptr nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_init(ptr nocapture nofree readonly, i32, i32, ptr)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_wait(ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_post(ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_fini(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_alloc(i32, i64, ptr)

; OPTIMISTIC: ; Function Attrs: nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_free(i32, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_init_allocator(i32, ptr, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_destroy_allocator(i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: write)
; OPTIMISTIC-NEXT: declare void @__kmpc_push_target_tripcount_mapper(ptr, i64, i64)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare i64 @__kmpc_warp_active_thread_mask()

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_syncwarp(i64)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_nowait_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_teams_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_teams_nowait_mapper(ptr, i64, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, ptr, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_register_requires(i64)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_begin_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_end_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_update_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_update_nowait_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i64 @__tgt_mapper_num_components(ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_push_mapper_component(ptr, ptr, ptr, i64, i64, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_task_allow_completion_event(ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_task_reduction_get_th_data(i32, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_task_reduction_init(i32, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_task_reduction_modifier_init(ptr, i32, i32, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_proxy_task_completed_ooo(ptr)

; OPTIMISTIC: ; Function Attrs: cold convergent noinline nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_barrier_simple_spmd(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_aligned_alloc(i32, i64, i64, ptr)

; OPTIMISTIC: ; Function Attrs: nosync nounwind allocsize(0)
; OPTIMISTIC-NEXT: declare noalias ptr @__kmpc_alloc_shared(i64)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC: declare void @__kmpc_barrier_simple_generic(ptr nocapture nofree readonly, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_begin_sharing_variables(ptr, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC: declare void @__kmpc_distribute_static_fini(ptr nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC: declare void @__kmpc_distribute_static_init_4(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC: declare void @__kmpc_distribute_static_init_4u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC: declare void @__kmpc_distribute_static_init_8(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC: declare void @__kmpc_distribute_static_init_8u(ptr nocapture nofree readonly, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC: declare void @__kmpc_end_masked(ptr nocapture nofree readonly, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_end_sharing_variables()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_error(ptr, i32, ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_fork_call_if(ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly, i32, ptr)

; OPTIMISTIC: ; Function Attrs: nosync nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_free_shared(ptr allocptr nocapture, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_get_hardware_num_blocks()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_get_hardware_num_threads_in_block()

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_get_hardware_thread_id_in_block()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_get_shared_variables(ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(inaccessiblemem: read)
; OPTIMISTIC-NEXT: declare i32 @__kmpc_get_warp_size()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare i8 @__kmpc_is_spmd_exec_mode()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_kernel_end_parallel()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare i1 @__kmpc_kernel_parallel(ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_kernel_prepare_parallel(ptr)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
; OPTIMISTIC: declare i32 @__kmpc_masked(ptr nocapture nofree readonly, i32, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_nvptx_end_reduce_nowait(i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(ptr, i32, i32, i64, ptr, ptr, ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare i32 @__kmpc_nvptx_teams_reduce_nowait_v2(ptr, i32, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC: nofree nosync nounwind willreturn
; OPTIMISTIC: declare i32 @__kmpc_omp_reg_task_with_affinity(ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly, i32, ptr nocapture nofree readonly)

; OPTIMISTIC: alwaysinline
; OPTIMISTIC: declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare i32 @__kmpc_shuffle_int32(i32, i16, i16)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare i64 @__kmpc_shuffle_int64(i64, i16, i16)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__kmpc_target_deinit(ptr, i8)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare i32 @__kmpc_target_init(ptr, i8, i1)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__tgt_interop_destroy(ptr, i32, ptr, i32, i32, ptr, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__tgt_interop_init(ptr, i32, ptr, i32, i32, i32, ptr, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__tgt_interop_use(ptr, i32, ptr, i32, i32, ptr, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__tgt_target_data_begin_mapper_issue(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare void @__tgt_target_data_begin_mapper_wait(i64, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC: declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC: declare i32 @__tgt_target_kernel_nowait(ptr, i64, i32, i32, ptr, ptr, i32, ptr, i32, ptr)

;;; Check extensions of integer params / retvals <= i32. Only functions in this file which are handled in OMPIRBuilder will get these attributes.
; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_set_num_threads(i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_set_dynamic(i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_set_nested(i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_set_max_active_levels(i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_set_schedule(i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_num_threads()

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @use_int(i32)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_dynamic()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_nested()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_max_threads()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_thread_num()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_num_procs()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_in_parallel()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_in_final()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_active_level()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_level()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_ancestor_thread_num(i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_team_size(i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_thread_limit()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_max_active_levels()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_get_schedule(ptr nocapture writeonly, ptr nocapture writeonly)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_max_task_priority()

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_init_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_set_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_unset_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_destroy_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_test_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_init_nest_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_set_nest_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_unset_nest_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_destroy_nest_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_test_nest_lock(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_init_lock_with_hint(ptr, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_init_nest_lock_with_hint(ptr, i32)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local double @omp_get_wtime()

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @use_double(double)

; EXT-NOT: Function Attrs
; EXT: declare dso_local double @omp_get_wtick()

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_default_device()

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_set_default_device(i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_is_initial_device()

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_num_devices()

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_num_teams()

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_team_num()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_cancellation()

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_initial_device()

; EXT-NOT: Function Attrs
; EXT: declare dso_local ptr @omp_target_alloc(i64, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @use_voidptr(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_target_free(ptr, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_target_is_present(ptr, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_target_memcpy(ptr, ptr, i64, i64, i64, i32, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_target_associate_ptr(ptr, ptr, i64, i64, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_target_disassociate_ptr(ptr, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_device_num()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_proc_bind()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_num_places()

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_get_place_num_procs(i32)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_get_place_proc_ids(i32 signext, ptr nocapture writeonly)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_place_num()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_partition_num_places()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local void @omp_get_partition_place_nums(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_control_tool(i32, i32, ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_destroy_allocator(i64)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_set_default_allocator(i64)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i64 @omp_get_default_allocator()

; EXT-NOT: Function Attrs
; EXT: declare dso_local ptr @omp_alloc(i64, i64)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_free(ptr, i64)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @ompc_set_affinity_format(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i64 @ompc_get_affinity_format(ptr, i64)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @use_sizet(i64)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @ompc_display_affinity(ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i64 @ompc_capture_affinity(ptr, i64, ptr)

; EXT-NOT: Function Attrs
; EXT: declare dso_local void @omp_fulfill_event(i64)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_pause_resource(i32, i32)

; EXT-NOT: Function Attrs
; EXT: declare dso_local i32 @omp_pause_resource_all(i32)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare dso_local signext i32 @omp_get_supported_active_levels()

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_barrier(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_cancel(ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare signext i32 @__kmpc_cancel_barrier(ptr, i32 signext)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_flush(ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_global_thread_num(ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_fork_call(ptr, i32 signext, ptr, ...)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare signext i32 @__kmpc_omp_taskwait(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_omp_taskyield(ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_push_num_threads(ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_push_proc_bind(ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_serialized_parallel(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_end_serialized_parallel(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_master(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_end_master(ptr, i32 signext)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_critical(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_critical_with_hint(ptr, i32 signext, ptr, i32 zeroext)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_end_critical(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_begin(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_end(ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare signext i32 @__kmpc_reduce(ptr, i32 signext, i32 signext, i64, ptr, ptr, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare signext i32 @__kmpc_reduce_nowait(ptr, i32 signext, i32 signext, i64, ptr, ptr, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_end_reduce(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_end_reduce_nowait(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_ordered(ptr, i32 signext)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_end_ordered(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_for_static_init_4(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_for_static_init_4u(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_for_static_init_8(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_for_static_init_8u(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_for_static_fini(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_team_static_init_4(ptr, i32 signext, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_team_static_init_4u(ptr, i32 signext, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_team_static_init_8(ptr, i32 signext, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_team_static_init_8u(ptr, i32 signext, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_for_static_init_4(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_for_static_init_4u(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_for_static_init_8(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_for_static_init_8u(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare signext i32 @__kmpc_single(ptr, i32 signext)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_end_single(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_omp_task_alloc(ptr, i32 signext, i32 signext, i64, i64, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_omp_task(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_end_taskgroup(ptr, i32 signext)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_taskgroup(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_dispatch_init_4(ptr, i32 signext, i32 signext, ptr, i32 signext, i32 signext, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_dispatch_init_4u(ptr, i32 signext, i32 signext, ptr, i32 zeroext, i32 zeroext, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_dispatch_init_8(ptr, i32 signext, i32 signext, ptr, i64, i64, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dist_dispatch_init_8u(ptr, i32 signext, i32 signext, ptr, i64, i64, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_init_4(ptr, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_init_4u(ptr, i32 signext, i32 signext, i32 zeroext, i32 zeroext, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_init_8(ptr, i32 signext, i32 signext, i64, i64, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_init_8u(ptr, i32 signext, i32 signext, i64, i64, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_dispatch_next_4(ptr, i32 signext, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_dispatch_next_4u(ptr, i32 signext, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_dispatch_next_8(ptr, i32 signext, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_dispatch_next_8u(ptr, i32 signext, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_fini_4(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_fini_4u(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_fini_8(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_dispatch_fini_8u(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_omp_task_begin_if0(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_omp_task_complete_if0(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_omp_task_with_deps(ptr, i32 signext, ptr, i32 signext, ptr, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_omp_wait_deps(ptr, i32 signext, i32 signext, ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__kmpc_cancellationpoint(ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_push_num_teams(ptr, i32 signext, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_fork_teams(ptr, i32 signext, ptr, ...)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_taskloop(ptr, i32 signext, ptr, i32 signext, ptr, ptr, i64, i32 signext, i32 signext, i64, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_omp_target_task_alloc(ptr, i32 signext, i32 signext, i64, i64, ptr, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_taskred_modifier_init(ptr, i32 signext, i32 signext, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_taskred_init(i32 signext, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_task_reduction_modifier_fini(ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_copyprivate(ptr, i32 signext, i64, ptr, ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_threadprivate_cached(ptr, i32 signext, ptr, i64, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_threadprivate_register(ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_doacross_init(ptr, i32 signext, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_doacross_wait(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_doacross_post(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_doacross_fini(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_alloc(i32 signext, i64, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_free(i32 signext, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_init_allocator(i32 signext, ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_destroy_allocator(i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_push_target_tripcount_mapper(ptr, i64, i64)

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare i64 @__kmpc_warp_active_thread_mask()

; EXT: ; Function Attrs: convergent nounwind
; EXT-NEXT: declare void @__kmpc_syncwarp(i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__tgt_target_mapper(ptr, i64, ptr, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__tgt_target_nowait_mapper(ptr, i64, ptr, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr, i32 signext, ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__tgt_target_teams_mapper(ptr, i64, ptr, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare signext i32 @__tgt_target_teams_nowait_mapper(ptr, i64, ptr, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr, i32 signext, i32 signext, i32 signext, ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_register_requires(i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_target_data_begin_mapper(ptr, i64, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_target_data_begin_nowait_mapper(ptr, i64, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_target_data_end_mapper(ptr, i64, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_target_data_end_nowait_mapper(ptr, i64, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_target_data_update_mapper(ptr, i64, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_target_data_update_nowait_mapper(ptr, i64, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare i64 @__tgt_mapper_num_components(ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__tgt_push_mapper_component(ptr, ptr, ptr, i64, i64, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_task_allow_completion_event(ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_task_reduction_get_th_data(i32 signext, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_task_reduction_init(i32 signext, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_task_reduction_modifier_init(ptr, i32 signext, i32 signext, i32 signext, ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare void @__kmpc_proxy_task_completed_ooo(ptr)

; EXT: ; Function Attrs: cold convergent noinline nounwind
; EXT-NEXT: declare void @__kmpc_barrier_simple_spmd(ptr nocapture nofree readonly, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare ptr @__kmpc_aligned_alloc(i32 signext, i64, i64, ptr)

; EXT: ; Function Attrs: nosync nounwind allocsize(0)
; EXT-NEXT: declare ptr @__kmpc_alloc_shared(i64)

; EXT: ; Function Attrs: convergent nounwind
; EXT: declare void @__kmpc_barrier_simple_generic(ptr, i32 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_begin_sharing_variables(ptr, i64)

; EXT: ; Function Attrs: nounwind
; EXT: declare void @__kmpc_distribute_static_fini(ptr, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT: declare void @__kmpc_distribute_static_init_4(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT: declare void @__kmpc_distribute_static_init_4u(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i32 signext, i32 signext)

; EXT: ; Function Attrs: nounwind
; EXT: declare void @__kmpc_distribute_static_init_8(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT: declare void @__kmpc_distribute_static_init_8u(ptr, i32 signext, i32 signext, ptr, ptr, ptr, ptr, i64, i64)

; EXT: ; Function Attrs: nounwind
; EXT: declare void @__kmpc_end_masked(ptr, i32 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_end_sharing_variables()

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_error(ptr, i32 signext, ptr)

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_fork_call_if(ptr, i32 signext, ptr, i32 signext, ptr)

; EXT: ; Function Attrs: nosync nounwind
; EXT-NEXT: declare void @__kmpc_free_shared(ptr allocptr nocapture, i64)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare zeroext i32 @__kmpc_get_hardware_num_blocks()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare zeroext i32 @__kmpc_get_hardware_num_threads_in_block()

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare zeroext i32 @__kmpc_get_hardware_thread_id_in_block()

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_get_shared_variables(ptr)

; EXT: ; Function Attrs: nounwind
; EXT-NEXT: declare zeroext i32 @__kmpc_get_warp_size()

; EXT-NOT: Function Attrs
; EXT: declare signext i8 @__kmpc_is_spmd_exec_mode()

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_kernel_end_parallel()

; EXT-NOT: Function Attrs
; EXT: declare i1 @__kmpc_kernel_parallel(ptr)

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_kernel_prepare_parallel(ptr)

; EXT: ; Function Attrs: nounwind
; EXT: declare signext i32 @__kmpc_masked(ptr, i32 signext, i32 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_nvptx_end_reduce_nowait(i32 signext)

; EXT-NOT: Function Attrs
; EXT: declare signext i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(ptr, i32 signext, i32 signext, i64, ptr, ptr, ptr)

; EXT-NOT: Function Attrs
; EXT: declare signext i32 @__kmpc_nvptx_teams_reduce_nowait_v2(ptr, i32 signext, ptr, i32 zeroext, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT: declare signext i32 @__kmpc_omp_reg_task_with_affinity(ptr, i32 signext, ptr, i32 signext, ptr)

; EXT: ; Function Attrs: alwaysinline
; EXT: declare void @__kmpc_parallel_51(ptr, i32 signext, i32 signext, i32 signext, i32 signext, ptr, ptr, ptr, i64)

; EXT-NOT: Function Attrs
; EXT: declare signext i32 @__kmpc_shuffle_int32(i32 signext, i16 signext, i16 signext)

; EXT-NOT: Function Attrs
; EXT: declare i64 @__kmpc_shuffle_int64(i64, i16 signext, i16 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__kmpc_target_deinit(ptr, i8 signext)

; EXT-NOT: Function Attrs
; EXT: declare signext i32 @__kmpc_target_init(ptr, i8 signext, i1 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__tgt_interop_destroy(ptr, i32 signext, ptr, i32 signext, i32 signext, ptr, i32 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__tgt_interop_init(ptr, i32 signext, ptr, i32 signext, i32 signext, i32, ptr, i32 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__tgt_interop_use(ptr, i32 signext, ptr, i32 signext, i32 signext, ptr, i32 signext)

; EXT-NOT: Function Attrs
; EXT: declare void @__tgt_target_data_begin_mapper_issue(ptr, i64, i32 signext, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

; EXT-NOT: Function Attrs
; EXT: declare void @__tgt_target_data_begin_mapper_wait(i64, ptr)

; EXT: ; Function Attrs: nounwind
; EXT: declare signext i32 @__tgt_target_kernel(ptr, i64, i32 signext, i32 signext, ptr, ptr)

; EXT: ; Function Attrs: nounwind
; EXT: declare signext i32 @__tgt_target_kernel_nowait(ptr, i64, i32 signext, i32 signext, ptr, ptr, i32 signext, ptr, i32 signext, ptr)

; MIPS_EXT: ; Function Attrs: nounwind
; MIPS_EXT: declare dso_local void @omp_set_num_threads(i32 signext)

; MIPS_EXT: ; Function Attrs: nounwind
; MIPS_EXT: declare dso_local i32 @omp_get_num_threads()

; MIPS_EXT: ; Function Attrs: convergent nounwind
; MIPS_EXT: declare void @__kmpc_critical_with_hint(ptr, i32 signext, ptr, i32 signext)

; MIPS_EXT: ; Function Attrs: nounwind
; MIPS_EXT: declare i32 @__kmpc_get_hardware_num_blocks()

; RISCV_EXT: ; Function Attrs: nounwind
; RISCV_EXT: declare signext i32 @__kmpc_get_hardware_num_blocks()

; RISCV_EXT: ; Function Attrs: nounwind
; RISCV_EXT: declare signext i32 @__kmpc_get_hardware_num_threads_in_block()

; RISCV_EXT: ; Function Attrs: nounwind
; RISCV_EXT: declare signext i32 @__kmpc_get_hardware_thread_id_in_block()

; RISCV_EXT: ; Function Attrs: nounwind
; RISCV_EXT: declare signext i32 @__kmpc_get_warp_size()

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"openmp", i32 50}
