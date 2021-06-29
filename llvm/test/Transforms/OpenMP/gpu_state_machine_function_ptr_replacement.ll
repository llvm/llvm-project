; RUN: opt -S -passes=openmp-opt-cgscc -openmp-ir-builder-optimistic-attributes -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s
; RUN: opt -S -passes=openmp-opt-cgscc -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s
; RUN: opt -S        -openmp-opt-cgscc -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s

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

; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* noundef @1, i32 %1, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*)* @__omp_outlined__1 to i8*), i8* noundef @__omp_outlined__1_wrapper.ID, i8** noundef %2, i64 noundef 0)
; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %0, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** %1, i64 0)
; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* noundef @1, i32 %1, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*)* @__omp_outlined__3 to i8*), i8* noundef @__omp_outlined__3_wrapper.ID, i8** noundef %3, i64 noundef 0)


%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8

define internal void @__omp_offloading_50_6dfa0f01_foo_l6_worker()  {
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
  %4 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  %5 = load i8*, i8** %work_fn, align 8
  %work_match = icmp eq i8* %5, bitcast (void (i16, i32)* @__omp_outlined__1_wrapper to i8*)
  br i1 %work_match, label %.execute.fn, label %.check.next

.execute.fn:                                      ; preds = %.execute.parallel
  call void @__omp_outlined__1_wrapper(i16 zeroext 0, i32 %4) 
  br label %.terminate.parallel

.check.next:                                      ; preds = %.execute.parallel
  %6 = load i8*, i8** %work_fn, align 8
  %work_match1 = icmp eq i8* %6, bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*)
  br i1 %work_match1, label %.execute.fn2, label %.check.next3

.execute.fn2:                                     ; preds = %.check.next
  call void @__omp_outlined__2_wrapper(i16 zeroext 0, i32 %4) 
  br label %.terminate.parallel

.check.next3:                                     ; preds = %.check.next
  %7 = load i8*, i8** %work_fn, align 8
  %work_match4 = icmp eq i8* %7, bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*)
  br i1 %work_match4, label %.execute.fn5, label %.check.next6

.execute.fn5:                                     ; preds = %.check.next3
  call void @__omp_outlined__3_wrapper(i16 zeroext 0, i32 %4) 
  br label %.terminate.parallel

.check.next6:                                     ; preds = %.check.next3
  %8 = bitcast i8* %2 to void (i16, i32)*
  call void %8(i16 0, i32 %4)
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

define weak void @__omp_offloading_50_6dfa0f01_foo_l6()  {
entry:
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  store i32 0, i32* %.zero.addr, align 4
  %nvptx_tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %nvptx_num_threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %nvptx_warp_size = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %thread_limit = sub nuw i32 %nvptx_num_threads, %nvptx_warp_size
  %0 = icmp ult i32 %nvptx_tid, %thread_limit
  br i1 %0, label %.worker, label %.mastercheck

.worker:                                          ; preds = %entry
  call void @__omp_offloading_50_6dfa0f01_foo_l6_worker() 
  br label %.exit

.mastercheck:                                     ; preds = %entry
  %nvptx_tid1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %nvptx_num_threads2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %nvptx_warp_size3 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %1 = sub nuw i32 %nvptx_warp_size3, 1
  %2 = sub nuw i32 %nvptx_num_threads2, 1
  %3 = xor i32 %1, -1
  %master_tid = and i32 %2, %3
  %4 = icmp eq i32 %nvptx_tid1, %master_tid
  br i1 %4, label %.master, label %.exit

.master:                                          ; preds = %.mastercheck
  %nvptx_num_threads4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %nvptx_warp_size5 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %thread_limit6 = sub nuw i32 %nvptx_num_threads4, %nvptx_warp_size5
  call void @__kmpc_kernel_init(i32 %thread_limit6, i16 1)
  call void @__kmpc_data_sharing_init_stack()
  %5 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  store i32 %5, i32* %.threadid_temp., align 4
  call void @__omp_outlined__(i32* %.threadid_temp., i32* %.zero.addr) 
  br label %.termination.notifier

.termination.notifier:                            ; preds = %.master
  call void @__kmpc_kernel_deinit(i16 1)
  call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  br label %.exit

.exit:                                            ; preds = %.termination.notifier, %.mastercheck, %.worker
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() 

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() 

declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() 

declare void @__kmpc_kernel_init(i32, i16)

declare void @__kmpc_data_sharing_init_stack()

define internal void @__omp_outlined__(i32* noalias %.global_tid., i32* noalias %.bound_tid.)  {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %captured_vars_addrs = alloca [0 x i8*], align 8
  %captured_vars_addrs1 = alloca [0 x i8*], align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  %0 = load i32*, i32** %.global_tid..addr, align 8
  %1 = load i32, i32* %0, align 4
  %2 = bitcast [0 x i8*]* %captured_vars_addrs to i8**
  call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %1, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__1 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__1_wrapper to i8*), i8** %2, i64 0)
  call void @bar() 
  %3 = bitcast [0 x i8*]* %captured_vars_addrs1 to i8**
  call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %1, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__3 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*), i8** %3, i64 0)
  ret void
}

define internal void @__omp_outlined__1(i32* noalias %.global_tid., i32* noalias %.bound_tid.)  {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__1_wrapper(i16 zeroext %0, i32 %1)  {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__1(i32* %.addr1, i32* %.zero.addr) 
  ret void
}

declare void @__kmpc_get_shared_variables(i8***)

declare void @__kmpc_parallel_51(%struct.ident_t*, i32, i32, i32, i32, i8*, i8*, i8**, i64)

define hidden void @bar()  {
entry:
  %captured_vars_addrs = alloca [0 x i8*], align 8
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  %1 = bitcast [0 x i8*]* %captured_vars_addrs to i8**
  call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %0, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*)* @__omp_outlined__2 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8** %1, i64 0)
  ret void
}

define internal void @__omp_outlined__2(i32* noalias %.global_tid., i32* noalias %.bound_tid.)  {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__2_wrapper(i16 zeroext %0, i32 %1)  {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__2(i32* %.addr1, i32* %.zero.addr) 
  ret void
}

declare i32 @__kmpc_global_thread_num(%struct.ident_t*) 

define internal void @__omp_outlined__3(i32* noalias %.global_tid., i32* noalias %.bound_tid.)  {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

define internal void @__omp_outlined__3_wrapper(i16 zeroext %0, i32 %1)  {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8**, align 8
  store i32 0, i32* %.zero.addr, align 4
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  call void @__kmpc_get_shared_variables(i8*** %global_args)
  call void @__omp_outlined__3(i32* %.addr1, i32* %.zero.addr) 
  ret void
}

declare void @__kmpc_kernel_deinit(i16)

declare void @__kmpc_barrier_simple_spmd(%struct.ident_t*, i32) 

declare i1 @__kmpc_kernel_parallel(i8**)

declare void @__kmpc_kernel_end_parallel()


!nvvm.annotations = !{!1}
!llvm.module.flags = !{!2, !3}

!1 = !{void ()* @__omp_offloading_50_6dfa0f01_foo_l6, !"kernel", i32 1}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 7, !"openmp-device", i32 50}
