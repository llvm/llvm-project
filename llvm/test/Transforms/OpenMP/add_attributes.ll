; RUN: opt < %s -S -openmpopt        | FileCheck %s
; RUN: opt < %s -S -passes=openmpopt | FileCheck %s
; RUN: opt < %s -S -openmpopt        -openmp-ir-builder-optimistic-attributes | FileCheck %s --check-prefix=OPTIMISTIC
; RUN: opt < %s -S -passes=openmpopt -openmp-ir-builder-optimistic-attributes | FileCheck %s --check-prefix=OPTIMISTIC
;
; TODO: Not all omp_XXXX methods are known to the OpenMPIRBuilder/OpenMPOpt.
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.omp_lock_t = type { i8* }
%struct.omp_nest_lock_t = type { i8* }
%struct.ident_t = type { i32, i32, i32, i32, i8* }

define void @call_all(i32 %schedule, %struct.omp_lock_t* %lock, i32 %lock_hint, %struct.omp_nest_lock_t* %nest_lock, i32 %i, i8* %s, i64 %st, i8* %vp, double %d, i32 %proc_bind, i64 %allocator_handle, i8* %cp, i64 %event_handle, i32 %pause_resource) {
entry:
  %schedule.addr = alloca i32, align 4
  %lock.addr = alloca %struct.omp_lock_t*, align 8
  %lock_hint.addr = alloca i32, align 4
  %nest_lock.addr = alloca %struct.omp_nest_lock_t*, align 8
  %i.addr = alloca i32, align 4
  %s.addr = alloca i8*, align 8
  %st.addr = alloca i64, align 8
  %vp.addr = alloca i8*, align 8
  %d.addr = alloca double, align 8
  %proc_bind.addr = alloca i32, align 4
  %allocator_handle.addr = alloca i64, align 8
  %cp.addr = alloca i8*, align 8
  %event_handle.addr = alloca i64, align 8
  %pause_resource.addr = alloca i32, align 4
  store i32 %schedule, i32* %schedule.addr, align 4
  store %struct.omp_lock_t* %lock, %struct.omp_lock_t** %lock.addr, align 8
  store i32 %lock_hint, i32* %lock_hint.addr, align 4
  store %struct.omp_nest_lock_t* %nest_lock, %struct.omp_nest_lock_t** %nest_lock.addr, align 8
  store i32 %i, i32* %i.addr, align 4
  store i8* %s, i8** %s.addr, align 8
  store i64 %st, i64* %st.addr, align 8
  store i8* %vp, i8** %vp.addr, align 8
  store double %d, double* %d.addr, align 8
  store i32 %proc_bind, i32* %proc_bind.addr, align 4
  store i64 %allocator_handle, i64* %allocator_handle.addr, align 8
  store i8* %cp, i8** %cp.addr, align 8
  store i64 %event_handle, i64* %event_handle.addr, align 8
  store i32 %pause_resource, i32* %pause_resource.addr, align 4
  call void @omp_set_num_threads(i32 0)
  call void @omp_set_dynamic(i32 0)
  call void @omp_set_nested(i32 0)
  call void @omp_set_max_active_levels(i32 0)
  %0 = load i32, i32* %schedule.addr, align 4
  call void @omp_set_schedule(i32 %0, i32 0)
  %call = call i32 @omp_get_num_threads()
  store i32 %call, i32* %i.addr, align 4
  %1 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %1)
  %call1 = call i32 @omp_get_dynamic()
  store i32 %call1, i32* %i.addr, align 4
  %2 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %2)
  %call2 = call i32 @omp_get_nested()
  store i32 %call2, i32* %i.addr, align 4
  %3 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %3)
  %call3 = call i32 @omp_get_max_threads()
  store i32 %call3, i32* %i.addr, align 4
  %4 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %4)
  %call4 = call i32 @omp_get_thread_num()
  store i32 %call4, i32* %i.addr, align 4
  %5 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %5)
  %call5 = call i32 @omp_get_num_procs()
  store i32 %call5, i32* %i.addr, align 4
  %6 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %6)
  %call6 = call i32 @omp_in_parallel()
  store i32 %call6, i32* %i.addr, align 4
  %7 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %7)
  %call7 = call i32 @omp_in_final()
  store i32 %call7, i32* %i.addr, align 4
  %8 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %8)
  %call8 = call i32 @omp_get_active_level()
  store i32 %call8, i32* %i.addr, align 4
  %9 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %9)
  %call9 = call i32 @omp_get_level()
  store i32 %call9, i32* %i.addr, align 4
  %10 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %10)
  %call10 = call i32 @omp_get_ancestor_thread_num(i32 0)
  store i32 %call10, i32* %i.addr, align 4
  %11 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %11)
  %call11 = call i32 @omp_get_team_size(i32 0)
  store i32 %call11, i32* %i.addr, align 4
  %12 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %12)
  %call12 = call i32 @omp_get_thread_limit()
  store i32 %call12, i32* %i.addr, align 4
  %13 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %13)
  %call13 = call i32 @omp_get_max_active_levels()
  store i32 %call13, i32* %i.addr, align 4
  %14 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %14)
  call void @omp_get_schedule(i32* %schedule.addr, i32* %i.addr)
  %call14 = call i32 @omp_get_max_task_priority()
  store i32 %call14, i32* %i.addr, align 4
  %15 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %15)
  %16 = load %struct.omp_lock_t*, %struct.omp_lock_t** %lock.addr, align 8
  call void @omp_init_lock(%struct.omp_lock_t* %16)
  %17 = load %struct.omp_lock_t*, %struct.omp_lock_t** %lock.addr, align 8
  call void @omp_set_lock(%struct.omp_lock_t* %17)
  %18 = load %struct.omp_lock_t*, %struct.omp_lock_t** %lock.addr, align 8
  call void @omp_unset_lock(%struct.omp_lock_t* %18)
  %19 = load %struct.omp_lock_t*, %struct.omp_lock_t** %lock.addr, align 8
  call void @omp_destroy_lock(%struct.omp_lock_t* %19)
  %20 = load %struct.omp_lock_t*, %struct.omp_lock_t** %lock.addr, align 8
  %call15 = call i32 @omp_test_lock(%struct.omp_lock_t* %20)
  store i32 %call15, i32* %i.addr, align 4
  %21 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %21)
  %22 = load %struct.omp_nest_lock_t*, %struct.omp_nest_lock_t** %nest_lock.addr, align 8
  call void @omp_init_nest_lock(%struct.omp_nest_lock_t* %22)
  %23 = load %struct.omp_nest_lock_t*, %struct.omp_nest_lock_t** %nest_lock.addr, align 8
  call void @omp_set_nest_lock(%struct.omp_nest_lock_t* %23)
  %24 = load %struct.omp_nest_lock_t*, %struct.omp_nest_lock_t** %nest_lock.addr, align 8
  call void @omp_unset_nest_lock(%struct.omp_nest_lock_t* %24)
  %25 = load %struct.omp_nest_lock_t*, %struct.omp_nest_lock_t** %nest_lock.addr, align 8
  call void @omp_destroy_nest_lock(%struct.omp_nest_lock_t* %25)
  %26 = load %struct.omp_nest_lock_t*, %struct.omp_nest_lock_t** %nest_lock.addr, align 8
  %call16 = call i32 @omp_test_nest_lock(%struct.omp_nest_lock_t* %26)
  store i32 %call16, i32* %i.addr, align 4
  %27 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %27)
  %28 = load %struct.omp_lock_t*, %struct.omp_lock_t** %lock.addr, align 8
  %29 = load i32, i32* %lock_hint.addr, align 4
  call void @omp_init_lock_with_hint(%struct.omp_lock_t* %28, i32 %29)
  %30 = load %struct.omp_nest_lock_t*, %struct.omp_nest_lock_t** %nest_lock.addr, align 8
  %31 = load i32, i32* %lock_hint.addr, align 4
  call void @omp_init_nest_lock_with_hint(%struct.omp_nest_lock_t* %30, i32 %31)
  %call17 = call double @omp_get_wtime()
  store double %call17, double* %d.addr, align 8
  %32 = load double, double* %d.addr, align 8
  call void @use_double(double %32)
  %call18 = call double @omp_get_wtick()
  store double %call18, double* %d.addr, align 8
  %33 = load double, double* %d.addr, align 8
  call void @use_double(double %33)
  %call19 = call i32 @omp_get_default_device()
  store i32 %call19, i32* %i.addr, align 4
  %34 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %34)
  call void @omp_set_default_device(i32 0)
  %call20 = call i32 @omp_is_initial_device()
  store i32 %call20, i32* %i.addr, align 4
  %35 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %35)
  %call21 = call i32 @omp_get_num_devices()
  store i32 %call21, i32* %i.addr, align 4
  %36 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %36)
  %call22 = call i32 @omp_get_num_teams()
  store i32 %call22, i32* %i.addr, align 4
  %37 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %37)
  %call23 = call i32 @omp_get_team_num()
  store i32 %call23, i32* %i.addr, align 4
  %38 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %38)
  %call24 = call i32 @omp_get_cancellation()
  store i32 %call24, i32* %i.addr, align 4
  %39 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %39)
  %call25 = call i32 @omp_get_initial_device()
  store i32 %call25, i32* %i.addr, align 4
  %40 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %40)
  %41 = load i64, i64* %st.addr, align 8
  %42 = load i32, i32* %i.addr, align 4
  %call26 = call i8* @omp_target_alloc(i64 %41, i32 %42)
  store i8* %call26, i8** %vp.addr, align 8
  %43 = load i8*, i8** %vp.addr, align 8
  call void @use_voidptr(i8* %43)
  %44 = load i8*, i8** %vp.addr, align 8
  %45 = load i32, i32* %i.addr, align 4
  call void @omp_target_free(i8* %44, i32 %45)
  %46 = load i8*, i8** %vp.addr, align 8
  %47 = load i32, i32* %i.addr, align 4
  %call27 = call i32 @omp_target_is_present(i8* %46, i32 %47)
  store i32 %call27, i32* %i.addr, align 4
  %48 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %48)
  %49 = load i8*, i8** %vp.addr, align 8
  %50 = load i8*, i8** %vp.addr, align 8
  %51 = load i64, i64* %st.addr, align 8
  %52 = load i64, i64* %st.addr, align 8
  %53 = load i64, i64* %st.addr, align 8
  %54 = load i32, i32* %i.addr, align 4
  %55 = load i32, i32* %i.addr, align 4
  %call28 = call i32 @omp_target_memcpy(i8* %49, i8* %50, i64 %51, i64 %52, i64 %53, i32 %54, i32 %55)
  store i32 %call28, i32* %i.addr, align 4
  %56 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %56)
  %57 = load i8*, i8** %vp.addr, align 8
  %58 = load i8*, i8** %vp.addr, align 8
  %59 = load i64, i64* %st.addr, align 8
  %60 = load i64, i64* %st.addr, align 8
  %61 = load i32, i32* %i.addr, align 4
  %call29 = call i32 @omp_target_associate_ptr(i8* %57, i8* %58, i64 %59, i64 %60, i32 %61)
  store i32 %call29, i32* %i.addr, align 4
  %62 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %62)
  %63 = load i8*, i8** %vp.addr, align 8
  %64 = load i32, i32* %i.addr, align 4
  %call30 = call i32 @omp_target_disassociate_ptr(i8* %63, i32 %64)
  store i32 %call30, i32* %i.addr, align 4
  %65 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %65)
  %call31 = call i32 @omp_get_device_num()
  store i32 %call31, i32* %i.addr, align 4
  %66 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %66)
  %call32 = call i32 @omp_get_proc_bind()
  store i32 %call32, i32* %proc_bind.addr, align 4
  %call33 = call i32 @omp_get_num_places()
  store i32 %call33, i32* %i.addr, align 4
  %67 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %67)
  %call34 = call i32 @omp_get_place_num_procs(i32 0)
  store i32 %call34, i32* %i.addr, align 4
  %68 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %68)
  %69 = load i32, i32* %i.addr, align 4
  call void @omp_get_place_proc_ids(i32 %69, i32* %i.addr)
  %call35 = call i32 @omp_get_place_num()
  store i32 %call35, i32* %i.addr, align 4
  %70 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %70)
  %call36 = call i32 @omp_get_partition_num_places()
  store i32 %call36, i32* %i.addr, align 4
  %71 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %71)
  call void @omp_get_partition_place_nums(i32* %i.addr)
  %72 = load i32, i32* %i.addr, align 4
  %73 = load i32, i32* %i.addr, align 4
  %74 = load i8*, i8** %vp.addr, align 8
  %call37 = call i32 @omp_control_tool(i32 %72, i32 %73, i8* %74)
  store i32 %call37, i32* %i.addr, align 4
  %75 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %75)
  %76 = load i64, i64* %allocator_handle.addr, align 8
  call void @omp_destroy_allocator(i64 %76)
  %77 = load i64, i64* %allocator_handle.addr, align 8
  call void @omp_set_default_allocator(i64 %77)
  %call38 = call i64 @omp_get_default_allocator()
  store i64 %call38, i64* %allocator_handle.addr, align 8
  %78 = load i64, i64* %st.addr, align 8
  %79 = load i64, i64* %allocator_handle.addr, align 8
  %call39 = call i8* @omp_alloc(i64 %78, i64 %79)
  store i8* %call39, i8** %vp.addr, align 8
  %80 = load i8*, i8** %vp.addr, align 8
  call void @use_voidptr(i8* %80)
  %81 = load i8*, i8** %vp.addr, align 8
  %82 = load i64, i64* %allocator_handle.addr, align 8
  call void @omp_free(i8* %81, i64 %82)
  %83 = load i64, i64* %st.addr, align 8
  %84 = load i64, i64* %allocator_handle.addr, align 8
  %call40 = call i8* @omp_alloc(i64 %83, i64 %84)
  store i8* %call40, i8** %vp.addr, align 8
  %85 = load i8*, i8** %vp.addr, align 8
  call void @use_voidptr(i8* %85)
  %86 = load i8*, i8** %vp.addr, align 8
  %87 = load i64, i64* %allocator_handle.addr, align 8
  call void @omp_free(i8* %86, i64 %87)
  %88 = load i8*, i8** %s.addr, align 8
  call void @ompc_set_affinity_format(i8* %88)
  %89 = load i8*, i8** %cp.addr, align 8
  %90 = load i64, i64* %st.addr, align 8
  %call41 = call i64 @ompc_get_affinity_format(i8* %89, i64 %90)
  store i64 %call41, i64* %st.addr, align 8
  %91 = load i64, i64* %st.addr, align 8
  call void @use_sizet(i64 %91)
  %92 = load i8*, i8** %s.addr, align 8
  call void @ompc_display_affinity(i8* %92)
  %93 = load i8*, i8** %cp.addr, align 8
  %94 = load i64, i64* %st.addr, align 8
  %95 = load i8*, i8** %s.addr, align 8
  %call42 = call i64 @ompc_capture_affinity(i8* %93, i64 %94, i8* %95)
  store i64 %call42, i64* %st.addr, align 8
  %96 = load i64, i64* %st.addr, align 8
  call void @use_sizet(i64 %96)
  %97 = load i64, i64* %event_handle.addr, align 8
  call void @omp_fulfill_event(i64 %97)
  %98 = load i32, i32* %pause_resource.addr, align 4
  %99 = load i32, i32* %i.addr, align 4
  %call43 = call i32 @omp_pause_resource(i32 %98, i32 %99)
  store i32 %call43, i32* %i.addr, align 4
  %100 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %100)
  %101 = load i32, i32* %pause_resource.addr, align 4
  %call44 = call i32 @omp_pause_resource_all(i32 %101)
  store i32 %call44, i32* %i.addr, align 4
  %102 = load i32, i32* %i.addr, align 4
  call void @use_int(i32 %102)
  %call45 = call i32 @omp_get_supported_active_levels()
  store i32 %call45, i32* %i.addr, align 4
  %103 = load i32, i32* %i.addr, align 4
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

declare dso_local void @omp_get_schedule(i32*, i32*)

declare dso_local i32 @omp_get_max_task_priority()

declare dso_local void @omp_init_lock(%struct.omp_lock_t*)

declare dso_local void @omp_set_lock(%struct.omp_lock_t*)

declare dso_local void @omp_unset_lock(%struct.omp_lock_t*)

declare dso_local void @omp_destroy_lock(%struct.omp_lock_t*)

declare dso_local i32 @omp_test_lock(%struct.omp_lock_t*)

declare dso_local void @omp_init_nest_lock(%struct.omp_nest_lock_t*)

declare dso_local void @omp_set_nest_lock(%struct.omp_nest_lock_t*)

declare dso_local void @omp_unset_nest_lock(%struct.omp_nest_lock_t*)

declare dso_local void @omp_destroy_nest_lock(%struct.omp_nest_lock_t*)

declare dso_local i32 @omp_test_nest_lock(%struct.omp_nest_lock_t*)

declare dso_local void @omp_init_lock_with_hint(%struct.omp_lock_t*, i32)

declare dso_local void @omp_init_nest_lock_with_hint(%struct.omp_nest_lock_t*, i32)

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

declare dso_local i8* @omp_target_alloc(i64, i32)

declare dso_local void @use_voidptr(i8*)

declare dso_local void @omp_target_free(i8*, i32)

declare dso_local i32 @omp_target_is_present(i8*, i32)

declare dso_local i32 @omp_target_memcpy(i8*, i8*, i64, i64, i64, i32, i32)

declare dso_local i32 @omp_target_associate_ptr(i8*, i8*, i64, i64, i32)

declare dso_local i32 @omp_target_disassociate_ptr(i8*, i32)

declare dso_local i32 @omp_get_device_num()

declare dso_local i32 @omp_get_proc_bind()

declare dso_local i32 @omp_get_num_places()

declare dso_local i32 @omp_get_place_num_procs(i32)

declare dso_local void @omp_get_place_proc_ids(i32, i32*)

declare dso_local i32 @omp_get_place_num()

declare dso_local i32 @omp_get_partition_num_places()

declare dso_local void @omp_get_partition_place_nums(i32*)

declare dso_local i32 @omp_control_tool(i32, i32, i8*)

declare dso_local void @omp_destroy_allocator(i64)

declare dso_local void @omp_set_default_allocator(i64)

declare dso_local i64 @omp_get_default_allocator()

declare dso_local i8* @omp_alloc(i64, i64)

declare dso_local void @omp_free(i8*, i64)

declare dso_local void @ompc_set_affinity_format(i8*)

declare dso_local i64 @ompc_get_affinity_format(i8*, i64)

declare dso_local void @use_sizet(i64)

declare dso_local void @ompc_display_affinity(i8*)

declare dso_local i64 @ompc_capture_affinity(i8*, i64, i8*)

declare dso_local void @omp_fulfill_event(i64)

declare dso_local i32 @omp_pause_resource(i32, i32)

declare dso_local i32 @omp_pause_resource_all(i32)

declare dso_local i32 @omp_get_supported_active_levels()

declare void @__kmpc_barrier(%struct.ident_t*, i32)

declare i32 @__kmpc_cancel(%struct.ident_t*, i32, i32)

declare i32 @__kmpc_cancel_barrier(%struct.ident_t*, i32)

declare void @__kmpc_flush(%struct.ident_t*)

declare i32 @__kmpc_global_thread_num(%struct.ident_t*)

declare void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

declare i32 @__kmpc_omp_taskwait(%struct.ident_t*, i32)

declare i32 @__kmpc_omp_taskyield(%struct.ident_t*, i32, i32)

declare void @__kmpc_push_num_threads(%struct.ident_t*, i32, i32)

declare void @__kmpc_push_proc_bind(%struct.ident_t*, i32, i32)

declare void @__kmpc_serialized_parallel(%struct.ident_t*, i32)

declare void @__kmpc_end_serialized_parallel(%struct.ident_t*, i32)

declare i32 @__kmpc_master(%struct.ident_t*, i32)

declare void @__kmpc_end_master(%struct.ident_t*, i32)

declare void @__kmpc_critical(%struct.ident_t*, i32, [8 x i32]*)

declare void @__kmpc_critical_with_hint(%struct.ident_t*, i32, [8 x i32]*, i32)

declare void @__kmpc_end_critical(%struct.ident_t*, i32, [8 x i32]*)

declare void @__kmpc_begin(%struct.ident_t*, i32)

declare void @__kmpc_end(%struct.ident_t*)

declare i32 @__kmpc_reduce(%struct.ident_t*, i32, i32, i64, i8*, void (i8*, i8*)*, [8 x i32]*)

declare i32 @__kmpc_reduce_nowait(%struct.ident_t*, i32, i32, i64, i8*, void (i8*, i8*)*, [8 x i32]*)

declare void @__kmpc_end_reduce(%struct.ident_t*, i32, [8 x i32]*)

declare void @__kmpc_end_reduce_nowait(%struct.ident_t*, i32, [8 x i32]*)

declare void @__kmpc_ordered(%struct.ident_t*, i32)

declare void @__kmpc_end_ordered(%struct.ident_t*, i32)

declare void @__kmpc_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32)

declare void @__kmpc_for_static_init_4u(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32)

declare void @__kmpc_for_static_init_8(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64)

declare void @__kmpc_for_static_init_8u(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64)

declare void @__kmpc_for_static_fini(%struct.ident_t*, i32)

declare void @__kmpc_team_static_init_4(%struct.ident_t*, i32, i32*, i32*, i32*, i32*, i32, i32)

declare void @__kmpc_team_static_init_4u(%struct.ident_t*, i32, i32*, i32*, i32*, i32*, i32, i32)

declare void @__kmpc_team_static_init_8(%struct.ident_t*, i32, i32*, i64*, i64*, i64*, i64, i64)

declare void @__kmpc_team_static_init_8u(%struct.ident_t*, i32, i32*, i64*, i64*, i64*, i64, i64)

declare void @__kmpc_dist_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32*, i32, i32)

declare void @__kmpc_dist_for_static_init_4u(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32*, i32, i32)

declare void @__kmpc_dist_for_static_init_8(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64*, i64, i64)

declare void @__kmpc_dist_for_static_init_8u(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64*, i64, i64)

declare i32 @__kmpc_single(%struct.ident_t*, i32)

declare void @__kmpc_end_single(%struct.ident_t*, i32)

declare i8* @__kmpc_omp_task_alloc(%struct.ident_t*, i32, i32, i64, i64, i32 (i32, i8*)*)

declare i32 @__kmpc_omp_task(%struct.ident_t*, i32, i8*)

declare void @__kmpc_end_taskgroup(%struct.ident_t*, i32)

declare void @__kmpc_taskgroup(%struct.ident_t*, i32)

declare void @__kmpc_dist_dispatch_init_4(%struct.ident_t*, i32, i32, i32*, i32, i32, i32, i32)

declare void @__kmpc_dist_dispatch_init_4u(%struct.ident_t*, i32, i32, i32*, i32, i32, i32, i32)

declare void @__kmpc_dist_dispatch_init_8(%struct.ident_t*, i32, i32, i32*, i64, i64, i64, i64)

declare void @__kmpc_dist_dispatch_init_8u(%struct.ident_t*, i32, i32, i32*, i64, i64, i64, i64)

declare void @__kmpc_dispatch_init_4(%struct.ident_t*, i32, i32, i32, i32, i32, i32)

declare void @__kmpc_dispatch_init_4u(%struct.ident_t*, i32, i32, i32, i32, i32, i32)

declare void @__kmpc_dispatch_init_8(%struct.ident_t*, i32, i32, i64, i64, i64, i64)

declare void @__kmpc_dispatch_init_8u(%struct.ident_t*, i32, i32, i64, i64, i64, i64)

declare i32 @__kmpc_dispatch_next_4(%struct.ident_t*, i32, i32*, i32*, i32*, i32*)

declare i32 @__kmpc_dispatch_next_4u(%struct.ident_t*, i32, i32*, i32*, i32*, i32*)

declare i32 @__kmpc_dispatch_next_8(%struct.ident_t*, i32, i32*, i64*, i64*, i64*)

declare i32 @__kmpc_dispatch_next_8u(%struct.ident_t*, i32, i32*, i64*, i64*, i64*)

declare void @__kmpc_dispatch_fini_4(%struct.ident_t*, i32)

declare void @__kmpc_dispatch_fini_4u(%struct.ident_t*, i32)

declare void @__kmpc_dispatch_fini_8(%struct.ident_t*, i32)

declare void @__kmpc_dispatch_fini_8u(%struct.ident_t*, i32)

declare void @__kmpc_omp_task_begin_if0(%struct.ident_t*, i32, i8*)

declare void @__kmpc_omp_task_complete_if0(%struct.ident_t*, i32, i8*)

declare i32 @__kmpc_omp_task_with_deps(%struct.ident_t*, i32, i8*, i32, i8*, i32, i8*)

declare void @__kmpc_omp_wait_deps(%struct.ident_t*, i32, i32, i8*, i32, i8*)

declare i32 @__kmpc_cancellationpoint(%struct.ident_t*, i32, i32)

declare void @__kmpc_push_num_teams(%struct.ident_t*, i32, i32, i32)

declare void @__kmpc_fork_teams(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

declare void @__kmpc_taskloop(%struct.ident_t*, i32, i8*, i32, i64*, i64*, i64, i32, i32, i64, i8*)

declare i8* @__kmpc_omp_target_task_alloc(%struct.ident_t*, i32, i32, i64, i64, i32 (i32, i8*)*, i64)

declare i8* @__kmpc_taskred_modifier_init(%struct.ident_t*, i32, i32, i32, i8*)

declare i8* @__kmpc_taskred_init(i32, i32, i8*)

declare void @__kmpc_task_reduction_modifier_fini(%struct.ident_t*, i32, i32)

declare void @__kmpc_copyprivate(%struct.ident_t*, i32, i64, i8*, void (i8*, i8*)*, i32)

declare i8* @__kmpc_threadprivate_cached(%struct.ident_t*, i32, i8*, i64, i8***)

declare void @__kmpc_threadprivate_register(%struct.ident_t*, i8*, i8* (i8*)*, i8* (i8*, i8*)*, void (i8*)*)

declare void @__kmpc_doacross_init(%struct.ident_t*, i32, i32, i8*)

declare void @__kmpc_doacross_wait(%struct.ident_t*, i32, i64*)

declare void @__kmpc_doacross_post(%struct.ident_t*, i32, i64*)

declare void @__kmpc_doacross_fini(%struct.ident_t*, i32)

declare i8* @__kmpc_alloc(i32, i64, i8*)

declare void @__kmpc_free(i32, i8*, i8*)

declare i8* @__kmpc_init_allocator(i32, i8*, i32, i8*)

declare void @__kmpc_destroy_allocator(i32, i8*)

declare void @__kmpc_push_target_tripcount(i64, i64)

declare i32 @__tgt_target_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**)

declare i32 @__tgt_target_nowait_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**)

declare i32 @__tgt_target_teams_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**, i32, i32)

declare i32 @__tgt_target_teams_nowait_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**, i32, i32)

declare void @__tgt_register_requires(i64)

declare void @__tgt_target_data_begin_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

declare void @__tgt_target_data_begin_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

declare void @__tgt_target_data_end_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

declare void @__tgt_target_data_end_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

declare void @__tgt_target_data_update_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

declare void @__tgt_target_data_update_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

declare i64 @__tgt_mapper_num_components(i8*)

declare void @__tgt_push_mapper_component(i8*, i8*, i8*, i64, i64)

declare i8* @__kmpc_task_allow_completion_event(%struct.ident_t*, i32, i8*)

declare i8* @__kmpc_task_reduction_get_th_data(i32, i8*, i8*)

declare i8* @__kmpc_task_reduction_init(i32, i32, i8*)

declare i8* @__kmpc_task_reduction_modifier_init(i8*, i32, i32, i32, i8*)

declare void @__kmpc_proxy_task_completed_ooo(i8*)

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
; CHECK-NEXT: declare dso_local i32 @omp_get_num_threads() #0

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @use_int(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_dynamic() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_nested() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_max_threads() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_thread_num() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_num_procs() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_in_parallel() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_in_final() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_active_level() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_level() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_ancestor_thread_num(i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_team_size(i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_thread_limit() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_max_active_levels() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_get_schedule(i32* nocapture writeonly, i32* nocapture writeonly) #0

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_max_task_priority()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_lock(%struct.omp_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_set_lock(%struct.omp_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_unset_lock(%struct.omp_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_destroy_lock(%struct.omp_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_test_lock(%struct.omp_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_nest_lock(%struct.omp_nest_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_set_nest_lock(%struct.omp_nest_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_unset_nest_lock(%struct.omp_nest_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_destroy_nest_lock(%struct.omp_nest_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_test_nest_lock(%struct.omp_nest_lock_t*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_lock_with_hint(%struct.omp_lock_t*, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_init_nest_lock_with_hint(%struct.omp_nest_lock_t*, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local double @omp_get_wtime()

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
; CHECK-NEXT: declare dso_local i32 @omp_get_cancellation() #0

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_initial_device()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i8* @omp_target_alloc(i64, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @use_voidptr(i8*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_target_free(i8*, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_is_present(i8*, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_memcpy(i8*, i8*, i64, i64, i64, i32, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_associate_ptr(i8*, i8*, i64, i64, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_target_disassociate_ptr(i8*, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_device_num()

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_proc_bind() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_num_places() #0

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_get_place_num_procs(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_get_place_proc_ids(i32, i32* nocapture writeonly) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_place_num() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_partition_num_places() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local void @omp_get_partition_place_nums(i32*) #0

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_control_tool(i32, i32, i8*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_destroy_allocator(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_set_default_allocator(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i64 @omp_get_default_allocator()

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i8* @omp_alloc(i64, i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_free(i8*, i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @ompc_set_affinity_format(i8*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i64 @ompc_get_affinity_format(i8*, i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @use_sizet(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @ompc_display_affinity(i8*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i64 @ompc_capture_affinity(i8*, i64, i8*)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local void @omp_fulfill_event(i64)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_pause_resource(i32, i32)

; CHECK-NOT: Function Attrs
; CHECK: declare dso_local i32 @omp_pause_resource_all(i32)

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare dso_local i32 @omp_get_supported_active_levels() #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_barrier(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_cancel(%struct.ident_t*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_cancel_barrier(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_flush(%struct.ident_t*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_global_thread_num(%struct.ident_t*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_taskwait(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_taskyield(%struct.ident_t*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_num_threads(%struct.ident_t*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_proc_bind(%struct.ident_t*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_serialized_parallel(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_serialized_parallel(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_master(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_master(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_critical(%struct.ident_t*, i32, [8 x i32]*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_critical_with_hint(%struct.ident_t*, i32, [8 x i32]*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_critical(%struct.ident_t*, i32, [8 x i32]*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_begin(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end(%struct.ident_t*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_reduce(%struct.ident_t*, i32, i32, i64, i8*, void (i8*, i8*)*, [8 x i32]*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_reduce_nowait(%struct.ident_t*, i32, i32, i64, i8*, void (i8*, i8*)*, [8 x i32]*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_reduce(%struct.ident_t*, i32, [8 x i32]*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_reduce_nowait(%struct.ident_t*, i32, [8 x i32]*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_ordered(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_ordered(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_4u(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_8(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_init_8u(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_for_static_fini(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_4(%struct.ident_t*, i32, i32*, i32*, i32*, i32*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_4u(%struct.ident_t*, i32, i32*, i32*, i32*, i32*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_8(%struct.ident_t*, i32, i32*, i64*, i64*, i64*, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_team_static_init_8u(%struct.ident_t*, i32, i32*, i64*, i64*, i64*, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_4u(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_8(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64*, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_for_static_init_8u(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64*, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_single(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_single(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_omp_task_alloc(%struct.ident_t*, i32, i32, i64, i64, i32 (i32, i8*)*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_task(%struct.ident_t*, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_end_taskgroup(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_taskgroup(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_4(%struct.ident_t*, i32, i32, i32*, i32, i32, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_4u(%struct.ident_t*, i32, i32, i32*, i32, i32, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_8(%struct.ident_t*, i32, i32, i32*, i64, i64, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dist_dispatch_init_8u(%struct.ident_t*, i32, i32, i32*, i64, i64, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_4(%struct.ident_t*, i32, i32, i32, i32, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_4u(%struct.ident_t*, i32, i32, i32, i32, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_8(%struct.ident_t*, i32, i32, i64, i64, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_init_8u(%struct.ident_t*, i32, i32, i64, i64, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_4(%struct.ident_t*, i32, i32*, i32*, i32*, i32*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_4u(%struct.ident_t*, i32, i32*, i32*, i32*, i32*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_8(%struct.ident_t*, i32, i32*, i64*, i64*, i64*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_dispatch_next_8u(%struct.ident_t*, i32, i32*, i64*, i64*, i64*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_4(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_4u(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_8(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_dispatch_fini_8u(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_omp_task_begin_if0(%struct.ident_t*, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_omp_task_complete_if0(%struct.ident_t*, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_omp_task_with_deps(%struct.ident_t*, i32, i8*, i32, i8*, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_omp_wait_deps(%struct.ident_t*, i32, i32, i8*, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__kmpc_cancellationpoint(%struct.ident_t*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_num_teams(%struct.ident_t*, i32, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_fork_teams(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_taskloop(%struct.ident_t*, i32, i8*, i32, i64*, i64*, i64, i32, i32, i64, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_omp_target_task_alloc(%struct.ident_t*, i32, i32, i64, i64, i32 (i32, i8*)*, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_taskred_modifier_init(%struct.ident_t*, i32, i32, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_taskred_init(i32, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_task_reduction_modifier_fini(%struct.ident_t*, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_copyprivate(%struct.ident_t*, i32, i64, i8*, void (i8*, i8*)*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_threadprivate_cached(%struct.ident_t*, i32, i8*, i64, i8***) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_threadprivate_register(%struct.ident_t*, i8*, i8* (i8*)*, i8* (i8*, i8*)*, void (i8*)*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_init(%struct.ident_t*, i32, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_wait(%struct.ident_t*, i32, i64*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_post(%struct.ident_t*, i32, i64*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_doacross_fini(%struct.ident_t*, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_alloc(i32, i64, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_free(i32, i8*, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_init_allocator(i32, i8*, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_destroy_allocator(i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_push_target_tripcount(i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_nowait_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_teams_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i32 @__tgt_target_teams_nowait_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**, i32, i32) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_register_requires(i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_begin_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_begin_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_end_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_end_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_update_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_target_data_update_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i64 @__tgt_mapper_num_components(i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__tgt_push_mapper_component(i8*, i8*, i8*, i64, i64) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_task_allow_completion_event(%struct.ident_t*, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_task_reduction_get_th_data(i32, i8*, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_task_reduction_init(i32, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare i8* @__kmpc_task_reduction_modifier_init(i8*, i32, i32, i32, i8*) #0

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: declare void @__kmpc_proxy_task_completed_ooo(i8*) #0

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn writeonly
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_num_threads(i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn writeonly
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_dynamic(i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn writeonly
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_nested(i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn writeonly
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_max_active_levels(i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn writeonly
; OPTIMISTIC-NEXT: declare dso_local void @omp_set_schedule(i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_num_threads() #1

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @use_int(i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_dynamic() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_nested() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_max_threads() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_thread_num() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_num_procs() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_in_parallel() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_in_final() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_active_level() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_level() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_ancestor_thread_num(i32) #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_team_size(i32) #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_thread_limit() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_max_active_levels() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare dso_local void @omp_get_schedule(i32* nocapture writeonly, i32* nocapture writeonly) #2

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_max_task_priority()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_lock(%struct.omp_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_set_lock(%struct.omp_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_unset_lock(%struct.omp_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_destroy_lock(%struct.omp_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_test_lock(%struct.omp_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_nest_lock(%struct.omp_nest_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_set_nest_lock(%struct.omp_nest_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_unset_nest_lock(%struct.omp_nest_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_destroy_nest_lock(%struct.omp_nest_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_test_nest_lock(%struct.omp_nest_lock_t*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_lock_with_hint(%struct.omp_lock_t*, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_init_nest_lock_with_hint(%struct.omp_nest_lock_t*, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local double @omp_get_wtime()

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

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_cancellation() #1

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_initial_device()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i8* @omp_target_alloc(i64, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @use_voidptr(i8*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_target_free(i8*, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_is_present(i8*, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_memcpy(i8*, i8*, i64, i64, i64, i32, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_associate_ptr(i8*, i8*, i64, i64, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_target_disassociate_ptr(i8*, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_device_num()

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_proc_bind() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_num_places() #1

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_get_place_num_procs(i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind
; OPTIMISTIC-NEXT: declare dso_local void @omp_get_place_proc_ids(i32, i32* nocapture writeonly) #2

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_place_num() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_partition_num_places() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly
; OPTIMISTIC-NEXT: declare dso_local void @omp_get_partition_place_nums(i32*) #1

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_control_tool(i32, i32, i8*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_destroy_allocator(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_set_default_allocator(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i64 @omp_get_default_allocator()

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i8* @omp_alloc(i64, i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_free(i8*, i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @ompc_set_affinity_format(i8*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i64 @ompc_get_affinity_format(i8*, i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @use_sizet(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @ompc_display_affinity(i8*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i64 @ompc_capture_affinity(i8*, i64, i8*)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local void @omp_fulfill_event(i64)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_pause_resource(i32, i32)

; OPTIMISTIC-NOT: Function Attrs
; OPTIMISTIC: declare dso_local i32 @omp_pause_resource_all(i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare dso_local i32 @omp_get_supported_active_levels() #1

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind readonly willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_global_thread_num(%struct.ident_t* nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_fork_call(%struct.ident_t* nocapture nofree readonly, i32, void (i32*, i32*, ...)* nocapture nofree readonly, ...)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_taskwait(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_taskyield(%struct.ident_t* nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_push_num_threads(%struct.ident_t* nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_push_proc_bind(%struct.ident_t* nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_serialized_parallel(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_end_serialized_parallel(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_master(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_end_master(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_critical(%struct.ident_t* nocapture nofree readonly, i32, [8 x i32]*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_critical_with_hint(%struct.ident_t* nocapture nofree readonly, i32, [8 x i32]*, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_critical(%struct.ident_t* nocapture nofree readonly, i32, [8 x i32]*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_begin(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_end(%struct.ident_t* nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_reduce(%struct.ident_t* nocapture nofree readonly, i32, i32, i64, i8* nocapture nofree readonly, void (i8*, i8*)*, [8 x i32]*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_reduce_nowait(%struct.ident_t* nocapture nofree readonly, i32, i32, i64, i8* nocapture nofree readonly, void (i8*, i8*)*, [8 x i32]*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_reduce(%struct.ident_t* nocapture nofree readonly, i32, [8 x i32]*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_reduce_nowait(%struct.ident_t* nocapture nofree readonly, i32, [8 x i32]*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_ordered(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_ordered(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_4(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_4u(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_8(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_init_8u(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_for_static_fini(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_4(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_4u(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_8(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_team_static_init_8u(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_4(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_4u(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_8(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_for_static_init_8u(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64, i64)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__kmpc_single(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_single(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_omp_task_alloc(%struct.ident_t* nocapture nofree readonly, i32, i32, i64, i64, i32 (i32, i8*)* nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_task(%struct.ident_t* nocapture nofree readonly, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_end_taskgroup(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_taskgroup(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_4(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_4u(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_8(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dist_dispatch_init_8u(%struct.ident_t* nocapture nofree readonly, i32, i32, i32* nocapture nofree, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_4(%struct.ident_t* nocapture nofree readonly, i32, i32, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_4u(%struct.ident_t* nocapture nofree readonly, i32, i32, i32, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_8(%struct.ident_t* nocapture nofree readonly, i32, i32, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_init_8u(%struct.ident_t* nocapture nofree readonly, i32, i32, i64, i64, i64, i64)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_4(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_4u(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree, i32* nocapture nofree)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_8(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_dispatch_next_8u(%struct.ident_t* nocapture nofree readonly, i32, i32* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree, i64* nocapture nofree)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_4(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_4u(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_8(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_dispatch_fini_8u(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_omp_task_begin_if0(%struct.ident_t* nocapture nofree readonly, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_omp_task_complete_if0(%struct.ident_t* nocapture nofree readonly, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_omp_task_with_deps(%struct.ident_t* nocapture nofree readonly, i32, i8*, i32, i8* nocapture nofree readonly, i32, i8* nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_omp_wait_deps(%struct.ident_t* nocapture nofree readonly, i32, i32, i8* nocapture nofree readonly, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i32 @__kmpc_cancellationpoint(%struct.ident_t* nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_push_num_teams(%struct.ident_t* nocapture nofree readonly, i32, i32, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_fork_teams(%struct.ident_t* nocapture nofree readonly, i32, void (i32*, i32*, ...)* nocapture nofree readonly, ...)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_taskloop(%struct.ident_t* nocapture nofree readonly, i32, i8*, i32, i64* nocapture nofree, i64* nocapture nofree, i64, i32, i32, i64, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_omp_target_task_alloc(%struct.ident_t* nocapture nofree readonly, i32, i32, i64, i64, i32 (i32, i8*)* nocapture nofree readonly, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_taskred_modifier_init(%struct.ident_t* nocapture nofree readonly, i32, i32, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare i8* @__kmpc_taskred_init(i32, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_task_reduction_modifier_fini(%struct.ident_t* nocapture nofree readonly, i32, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_copyprivate(%struct.ident_t* nocapture nofree readonly, i32, i64, i8* nocapture nofree readonly, void (i8*, i8*)*, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_threadprivate_cached(%struct.ident_t* nocapture nofree readonly, i32, i8*, i64, i8***)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_threadprivate_register(%struct.ident_t* nocapture nofree readonly, i8*, i8* (i8*)* nocapture nofree readonly, i8* (i8*, i8*)* nocapture nofree readonly, void (i8*)* nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_init(%struct.ident_t* nocapture nofree readonly, i32, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_wait(%struct.ident_t* nocapture nofree readonly, i32, i64* nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_post(%struct.ident_t* nocapture nofree readonly, i32, i64* nocapture nofree readonly)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_doacross_fini(%struct.ident_t* nocapture nofree readonly, i32)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_alloc(i32, i64, i8*)

; OPTIMISTIC: ; Function Attrs: nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_free(i32, i8*, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_init_allocator(i32, i8*, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_destroy_allocator(i32, i8*)

; OPTIMISTIC: ; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn writeonly
; OPTIMISTIC-NEXT: declare void @__kmpc_push_target_tripcount(i64, i64)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_nowait_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_teams_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**, i32, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i32 @__tgt_target_teams_nowait_mapper(i64, i8*, i32, i8**, i8**, i64*, i64*, i8**, i32, i32)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_register_requires(i64)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_begin_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_begin_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_end_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_end_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_update_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_target_data_update_nowait_mapper(i64, i32, i8**, i8**, i64*, i64*, i8**)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare i64 @__tgt_mapper_num_components(i8*)

; OPTIMISTIC: ; Function Attrs: nounwind
; OPTIMISTIC-NEXT: declare void @__tgt_push_mapper_component(i8*, i8*, i8*, i64, i64)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_task_allow_completion_event(%struct.ident_t* nocapture nofree readonly, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_task_reduction_get_th_data(i32, i8*, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_task_reduction_init(i32, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare noalias i8* @__kmpc_task_reduction_modifier_init(i8*, i32, i32, i32, i8*)

; OPTIMISTIC: ; Function Attrs: nofree nosync nounwind willreturn
; OPTIMISTIC-NEXT: declare void @__kmpc_proxy_task_completed_ooo(i8*)
