; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *

; CHECK-SPIRV-DAG: OpEntryPoint Kernel %[[#BlockKer1:]] "__device_side_enqueue_block_invoke_kernel"
; CHECK-SPIRV-DAG: OpEntryPoint Kernel %[[#BlockKer2:]] "__device_side_enqueue_block_invoke_2_kernel"
; CHECK-SPIRV-DAG: OpEntryPoint Kernel %[[#BlockKer3:]] "__device_side_enqueue_block_invoke_3_kernel"
; CHECK-SPIRV-DAG: OpEntryPoint Kernel %[[#BlockKer4:]] "__device_side_enqueue_block_invoke_4_kernel"
; CHECK-SPIRV-DAG: OpEntryPoint Kernel %[[#BlockKer5:]] "__device_side_enqueue_block_invoke_5_kernel"
; CHECK-SPIRV-DAG: OpName %[[#BlockGlb1:]] "__block_literal_global"
; CHECK-SPIRV-DAG: OpName %[[#BlockGlb2:]] "__block_literal_global.1"

; CHECK-SPIRV-DAG: %[[#Int32Ty:]] = OpTypeInt 32
; CHECK-SPIRV-DAG: %[[#Int8Ty:]] = OpTypeInt 8
; CHECK-SPIRV-DAG: %[[#VoidTy:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#Int8PtrGenTy:]] = OpTypePointer Generic %[[#Int8Ty]]
; CHECK-SPIRV-DAG: %[[#EventTy:]] = OpTypeDeviceEvent
; CHECK-SPIRV-DAG: %[[#EventPtrTy:]] = OpTypePointer Generic %[[#EventTy]]
; CHECK-SPIRV-DAG: %[[#Int32LocPtrTy:]] = OpTypePointer Function %[[#Int32Ty]]
; CHECK-SPIRV-DAG: %[[#BlockStructTy:]] = OpTypeStruct
; CHECK-SPIRV-DAG: %[[#BlockStructLocPtrTy:]] = OpTypePointer Function %[[#BlockStructTy]]
; CHECK-SPIRV-DAG: %[[#BlockTy1:]] = OpTypeFunction %[[#VoidTy]] %[[#Int8PtrGenTy]]
; CHECK-SPIRV-DAG: %[[#BlockTy2:]] = OpTypeFunction %[[#VoidTy]] %[[#Int8PtrGenTy]]
; CHECK-SPIRV-DAG: %[[#BlockTy3:]] = OpTypeFunction %[[#VoidTy]] %[[#Int8PtrGenTy]]

; CHECK-SPIRV-DAG: %[[#ConstInt0:]] = OpConstantNull %[[#Int32Ty]]
; CHECK-SPIRV-DAG: %[[#EventNull:]] = OpConstantNull %[[#EventPtrTy]]
; CHECK-SPIRV-DAG: %[[#ConstInt21:]] = OpConstant %[[#Int32Ty]] 21{{$}}
; CHECK-SPIRV-DAG: %[[#ConstInt8:]] = OpConstant %[[#Int32Ty]] 8{{$}}
; CHECK-SPIRV-DAG: %[[#ConstInt24:]] = OpConstant %[[#Int32Ty]] 24{{$}}
; CHECK-SPIRV-DAG: %[[#ConstInt12:]] = OpConstant %[[#Int32Ty]] 12{{$}}
; CHECK-SPIRV-DAG: %[[#ConstInt2:]] = OpConstant %[[#Int32Ty]] 2{{$}}

;; typedef struct {int a;} ndrange_t;
;; #define NULL ((void*)0)

;; kernel void device_side_enqueue(global int *a, global int *b, int i, char c0) {
;;   queue_t default_queue;
;;   unsigned flags = 0;
;;   ndrange_t ndrange;
;;   clk_event_t clk_event;
;;   clk_event_t event_wait_list;
;;   clk_event_t event_wait_list2[] = {clk_event};

;; Emits block literal on stack and block kernel.

; CHECK-SPIRV:      %[[#BlockLitPtr1:]] = OpBitcast %[[#BlockStructLocPtrTy]]
; CHECK-SPIRV-NEXT: %[[#BlockLit1:]] = OpPtrCastToGeneric %[[#Int8PtrGenTy]] %[[#BlockLitPtr1]]
; CHECK-SPIRV-NEXT: %[[#]] = OpEnqueueKernel %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#ConstInt0]] %[[#EventNull]] %[[#EventNull]] %[[#BlockKer1]] %[[#BlockLit1]] %[[#ConstInt21]] %[[#ConstInt8]]

;;   enqueue_kernel(default_queue, flags, ndrange,
;;                  ^(void) {
;;                    a[i] = c0;
;;                  });

;; Emits block literal on stack and block kernel.

; CHECK-SPIRV:      %[[#Event1:]] = OpPtrCastToGeneric %[[#EventPtrTy]]
; CHECK-SPIRV:      %[[#Event2:]] = OpPtrCastToGeneric %[[#EventPtrTy]]
; CHECK-SPIRV:      %[[#BlockLitPtr2:]] = OpBitcast %[[#BlockStructLocPtrTy]]
; CHECK-SPIRV-NEXT: %[[#BlockLit2:]] = OpPtrCastToGeneric %[[#Int8PtrGenTy]] %[[#BlockLitPtr2]]
; CHECK-SPIRV-NEXT: %[[#]] = OpEnqueueKernel %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#ConstInt2]] %[[#Event1]] %[[#Event2]] %[[#BlockKer2]] %[[#BlockLit2]] %[[#ConstInt24]] %[[#ConstInt8]]

;;   enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
;;                  ^(void) {
;;                    a[i] = b[i];
;;                  });

;;   char c;
;; Emits global block literal and block kernel.

; CHECK-SPIRV: %[[#Event1:]] = OpPtrCastToGeneric %[[#EventPtrTy]]
; CHECK-SPIRV: %[[#Event2:]] = OpPtrCastToGeneric %[[#EventPtrTy]]
; CHECK-SPIRV: %[[#BlockLit3Tmp:]] = OpBitcast %[[#]] %[[#BlockGlb1]]
; CHECK-SPIRV: %[[#BlockLit3:]] = OpPtrCastToGeneric %[[#Int8PtrGenTy]] %[[#BlockLit3Tmp]]
; CHECK-SPIRV: %[[#LocalBuf31:]] = OpPtrAccessChain %[[#Int32LocPtrTy]]
; CHECK-SPIRV: %[[#]] = OpEnqueueKernel %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#ConstInt2]] %[[#Event1]] %[[#Event2]] %[[#BlockKer3]] %[[#BlockLit3]] %[[#ConstInt12]] %[[#ConstInt8]] %[[#LocalBuf31]]

;;   enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
;;                  ^(local void *p) {
;;                    return;
;;                  },
;;                  c);

;; Emits global block literal and block kernel.

; CHECK-SPIRV:      %[[#BlockLit4Tmp:]] = OpBitcast %[[#]] %[[#BlockGlb2]]
; CHECK-SPIRV:      %[[#BlockLit4:]] = OpPtrCastToGeneric %[[#Int8PtrGenTy]] %[[#BlockLit4Tmp]]
; CHECK-SPIRV:      %[[#LocalBuf41:]] = OpPtrAccessChain %[[#Int32LocPtrTy]]
; CHECK-SPIRV-NEXT: %[[#LocalBuf42:]] = OpPtrAccessChain %[[#Int32LocPtrTy]]
; CHECK-SPIRV-NEXT: %[[#LocalBuf43:]] = OpPtrAccessChain %[[#Int32LocPtrTy]]
; CHECK-SPIRV-NEXT: %[[#]] = OpEnqueueKernel %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#ConstInt0]] %[[#EventNull]] %[[#EventNull]] %[[#BlockKer4]] %[[#BlockLit4]] %[[#ConstInt12]] %[[#ConstInt8]] %[[#LocalBuf41]] %[[#LocalBuf42]] %[[#LocalBuf43]]

;;   enqueue_kernel(default_queue, flags, ndrange,
;;                  ^(local void *p1, local void *p2, local void *p3) {
;;                    return;
;;                  },
;;                  1, 2, 4);

;; Emits block literal on stack and block kernel.

; CHECK-SPIRV:      %[[#Event1:]] = OpPtrCastToGeneric %[[#EventPtrTy]]
; CHECK-SPIRV:      %[[#BlockLit5Tmp:]] = OpBitcast %[[#BlockStructLocPtrTy]]
; CHECK-SPIRV-NEXT: %[[#BlockLit5:]] = OpPtrCastToGeneric %[[#Int8PtrGenTy]] %[[#BlockLit5Tmp]]
; CHECK-SPIRV-NEXT: %[[#]] = OpEnqueueKernel %[[#Int32Ty]] %[[#]] %[[#]] %[[#]] %[[#ConstInt0]] %[[#EventNull]] %[[#Event1]] %[[#BlockKer5]] %[[#BlockLit5]] %[[#ConstInt24]] %[[#ConstInt8]]

;;   enqueue_kernel(default_queue, flags, ndrange, 0, NULL, &clk_event,
;;                  ^(void) {
;;                    a[i] = b[i];
;;                  });
;; }

; CHECK-SPIRV-DAG: %[[#BlockKer1]] = OpFunction %[[#VoidTy]] None %[[#BlockTy1]]
; CHECK-SPIRV-DAG: %[[#BlockKer2]] = OpFunction %[[#VoidTy]] None %[[#BlockTy1]]
; CHECK-SPIRV-DAG: %[[#BlockKer3]] = OpFunction %[[#VoidTy]] None %[[#BlockTy3]]
; CHECK-SPIRV-DAG: %[[#BlockKer4]] = OpFunction %[[#VoidTy]] None %[[#BlockTy2]]
; CHECK-SPIRV-DAG: %[[#BlockKer5]] = OpFunction %[[#VoidTy]] None %[[#BlockTy1]]

%opencl.queue_t = type opaque
%struct.ndrange_t = type { i32 }
%opencl.clk_event_t = type opaque
%struct.__opencl_block_literal_generic = type { i32, i32, ptr addrspace(4) }

@__block_literal_global = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 12, i32 4, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_3 to ptr addrspace(4)) }, align 4
@__block_literal_global.1 = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 12, i32 4, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_4 to ptr addrspace(4)) }, align 4

define dso_local spir_kernel void @device_side_enqueue(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %i, i8 noundef signext %c0) {
entry:
  %a.addr = alloca ptr addrspace(1), align 4
  %b.addr = alloca ptr addrspace(1), align 4
  %i.addr = alloca i32, align 4
  %c0.addr = alloca i8, align 1
  %default_queue = alloca ptr, align 4
  %flags = alloca i32, align 4
  %ndrange = alloca %struct.ndrange_t, align 4
  %clk_event = alloca ptr, align 4
  %event_wait_list = alloca ptr, align 4
  %event_wait_list2 = alloca [1 x ptr], align 4
  %tmp = alloca %struct.ndrange_t, align 4
  %block = alloca <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, align 4
  %tmp3 = alloca %struct.ndrange_t, align 4
  %block4 = alloca <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, align 4
  %c = alloca i8, align 1
  %tmp11 = alloca %struct.ndrange_t, align 4
  %block_sizes = alloca [1 x i32], align 4
  %tmp12 = alloca %struct.ndrange_t, align 4
  %block_sizes13 = alloca [3 x i32], align 4
  %tmp14 = alloca %struct.ndrange_t, align 4
  %block15 = alloca <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, align 4
  store ptr addrspace(1) %a, ptr %a.addr, align 4
  store ptr addrspace(1) %b, ptr %b.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  store i8 %c0, ptr %c0.addr, align 1
  store i32 0, ptr %flags, align 4
  %arrayinit.begin = getelementptr inbounds [1 x ptr], ptr %event_wait_list2, i32 0, i32 0
  %0 = load ptr, ptr %clk_event, align 4
  store ptr %0, ptr %arrayinit.begin, align 4
  %1 = load ptr, ptr %default_queue, align 4
  %2 = load i32, ptr %flags, align 4
  %3 = bitcast ptr %tmp to ptr
  %4 = bitcast ptr %ndrange to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %3, ptr align 4 %4, i32 4, i1 false)
  %block.size = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr %block, i32 0, i32 0
  store i32 21, ptr %block.size, align 4
  %block.align = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr %block, i32 0, i32 1
  store i32 4, ptr %block.align, align 4
  %block.invoke = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr %block, i32 0, i32 2
  store ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke to ptr addrspace(4)), ptr %block.invoke, align 4
  %block.captured = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr %block, i32 0, i32 3
  %5 = load ptr addrspace(1), ptr %a.addr, align 4
  store ptr addrspace(1) %5, ptr %block.captured, align 4
  %block.captured1 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr %block, i32 0, i32 4
  %6 = load i32, ptr %i.addr, align 4
  store i32 %6, ptr %block.captured1, align 4
  %block.captured2 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr %block, i32 0, i32 5
  %7 = load i8, ptr %c0.addr, align 1
  store i8 %7, ptr %block.captured2, align 4
  %8 = bitcast ptr %block to ptr
  %9 = addrspacecast ptr %8 to ptr addrspace(4)
  %10 = call spir_func i32 @__enqueue_kernel_basic(ptr %1, i32 %2, ptr byval(%struct.ndrange_t) %tmp, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_kernel to ptr addrspace(4)), ptr addrspace(4) %9)
  %11 = load ptr, ptr %default_queue, align 4
  %12 = load i32, ptr %flags, align 4
  %13 = bitcast ptr %tmp3 to ptr
  %14 = bitcast ptr %ndrange to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %13, ptr align 4 %14, i32 4, i1 false)
  %15 = addrspacecast ptr %event_wait_list to ptr addrspace(4)
  %16 = addrspacecast ptr %clk_event to ptr addrspace(4)
  %block.size5 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block4, i32 0, i32 0
  store i32 24, ptr %block.size5, align 4
  %block.align6 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block4, i32 0, i32 1
  store i32 4, ptr %block.align6, align 4
  %block.invoke7 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block4, i32 0, i32 2
  store ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_2 to ptr addrspace(4)), ptr %block.invoke7, align 4
  %block.captured8 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block4, i32 0, i32 3
  %17 = load ptr addrspace(1), ptr %a.addr, align 4
  store ptr addrspace(1) %17, ptr %block.captured8, align 4
  %block.captured9 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block4, i32 0, i32 4
  %18 = load i32, ptr %i.addr, align 4
  store i32 %18, ptr %block.captured9, align 4
  %block.captured10 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block4, i32 0, i32 5
  %19 = load ptr addrspace(1), ptr %b.addr, align 4
  store ptr addrspace(1) %19, ptr %block.captured10, align 4
  %20 = bitcast ptr %block4 to ptr
  %21 = addrspacecast ptr %20 to ptr addrspace(4)
  %22 = call spir_func i32 @__enqueue_kernel_basic_events(ptr %11, i32 %12, ptr %tmp3, i32 2, ptr addrspace(4) %15, ptr addrspace(4) %16, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_2_kernel to ptr addrspace(4)), ptr addrspace(4) %21)
  %23 = load ptr, ptr %default_queue, align 4
  %24 = load i32, ptr %flags, align 4
  %25 = bitcast ptr %tmp11 to ptr
  %26 = bitcast ptr %ndrange to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %25, ptr align 4 %26, i32 4, i1 false)
  %arraydecay = getelementptr inbounds [1 x ptr], ptr %event_wait_list2, i32 0, i32 0
  %27 = addrspacecast ptr %arraydecay to ptr addrspace(4)
  %28 = addrspacecast ptr %clk_event to ptr addrspace(4)
  %29 = getelementptr [1 x i32], ptr %block_sizes, i32 0, i32 0
  %30 = load i8, ptr %c, align 1
  %31 = zext i8 %30 to i32
  store i32 %31, ptr %29, align 4
  %32 = call spir_func i32 @__enqueue_kernel_events_varargs(ptr %23, i32 %24, ptr %tmp11, i32 2, ptr addrspace(4) %27, ptr addrspace(4) %28, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_3_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global to ptr addrspace(4)), i32 1, ptr %29)
  %33 = load ptr, ptr %default_queue, align 4
  %34 = load i32, ptr %flags, align 4
  %35 = bitcast ptr %tmp12 to ptr
  %36 = bitcast ptr %ndrange to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %35, ptr align 4 %36, i32 4, i1 false)
  %37 = getelementptr [3 x i32], ptr %block_sizes13, i32 0, i32 0
  store i32 1, ptr %37, align 4
  %38 = getelementptr [3 x i32], ptr %block_sizes13, i32 0, i32 1
  store i32 2, ptr %38, align 4
  %39 = getelementptr [3 x i32], ptr %block_sizes13, i32 0, i32 2
  store i32 4, ptr %39, align 4
  %40 = call spir_func i32 @__enqueue_kernel_varargs(ptr %33, i32 %34, ptr %tmp12, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_4_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global.1 to ptr addrspace(4)), i32 3, ptr %37)
  %41 = load ptr, ptr %default_queue, align 4
  %42 = load i32, ptr %flags, align 4
  %43 = bitcast ptr %tmp14 to ptr
  %44 = bitcast ptr %ndrange to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %43, ptr align 4 %44, i32 4, i1 false)
  %45 = addrspacecast ptr %clk_event to ptr addrspace(4)
  %block.size16 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block15, i32 0, i32 0
  store i32 24, ptr %block.size16, align 4
  %block.align17 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block15, i32 0, i32 1
  store i32 4, ptr %block.align17, align 4
  %block.invoke18 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block15, i32 0, i32 2
  store ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_5 to ptr addrspace(4)), ptr %block.invoke18, align 4
  %block.captured19 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block15, i32 0, i32 3
  %46 = load ptr addrspace(1), ptr %a.addr, align 4
  store ptr addrspace(1) %46, ptr %block.captured19, align 4
  %block.captured20 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block15, i32 0, i32 4
  %47 = load i32, ptr %i.addr, align 4
  store i32 %47, ptr %block.captured20, align 4
  %block.captured21 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr %block15, i32 0, i32 5
  %48 = load ptr addrspace(1), ptr %b.addr, align 4
  store ptr addrspace(1) %48, ptr %block.captured21, align 4
  %49 = bitcast ptr %block15 to ptr
  %50 = addrspacecast ptr %49 to ptr addrspace(4)
  %51 = call spir_func i32 @__enqueue_kernel_basic_events(ptr %41, i32 %42, ptr %tmp14, i32 0, ptr addrspace(4) null, ptr addrspace(4) %45, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_5_kernel to ptr addrspace(4)), ptr addrspace(4) %50)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)

define internal spir_func void @__device_side_enqueue_block_invoke(ptr addrspace(4) noundef %.block_descriptor) {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  %block.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  %block = bitcast ptr addrspace(4) %.block_descriptor to ptr addrspace(4)
  store ptr addrspace(4) %block, ptr %block.addr, align 4
  %block.capture.addr = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr addrspace(4) %block, i32 0, i32 5
  %0 = load i8, ptr addrspace(4) %block.capture.addr, align 4
  %conv = sext i8 %0 to i32
  %block.capture.addr1 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr addrspace(4) %block, i32 0, i32 3
  %1 = load ptr addrspace(1), ptr addrspace(4) %block.capture.addr1, align 4
  %block.capture.addr2 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, ptr addrspace(4) %block, i32 0, i32 4
  %2 = load i32, ptr addrspace(4) %block.capture.addr2, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %1, i32 %2
  store i32 %conv, ptr addrspace(1) %arrayidx, align 4
  ret void
}

define spir_kernel void @__device_side_enqueue_block_invoke_kernel(ptr addrspace(4) %0) {
entry:
  call spir_func void @__device_side_enqueue_block_invoke(ptr addrspace(4) %0)
  ret void
}

declare spir_func i32 @__enqueue_kernel_basic(ptr, i32, ptr, ptr addrspace(4), ptr addrspace(4))

define internal spir_func void @__device_side_enqueue_block_invoke_2(ptr addrspace(4) noundef %.block_descriptor) {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  %block.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  %block = bitcast ptr addrspace(4) %.block_descriptor to ptr addrspace(4)
  store ptr addrspace(4) %block, ptr %block.addr, align 4
  %block.capture.addr = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 5
  %0 = load ptr addrspace(1), ptr addrspace(4) %block.capture.addr, align 4
  %block.capture.addr1 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 4
  %1 = load i32, ptr addrspace(4) %block.capture.addr1, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %0, i32 %1
  %2 = load i32, ptr addrspace(1) %arrayidx, align 4
  %block.capture.addr2 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 3
  %3 = load ptr addrspace(1), ptr addrspace(4) %block.capture.addr2, align 4
  %block.capture.addr3 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 4
  %4 = load i32, ptr addrspace(4) %block.capture.addr3, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr addrspace(1) %3, i32 %4
  store i32 %2, ptr addrspace(1) %arrayidx4, align 4
  ret void
}

define spir_kernel void @__device_side_enqueue_block_invoke_2_kernel(ptr addrspace(4) %0) {
entry:
  call spir_func void @__device_side_enqueue_block_invoke_2(ptr addrspace(4) %0)
  ret void
}

declare spir_func i32 @__enqueue_kernel_basic_events(ptr, i32, ptr, i32, ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4))

define internal spir_func void @__device_side_enqueue_block_invoke_3(ptr addrspace(4) noundef %.block_descriptor, ptr addrspace(3) noundef %p) {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  %p.addr = alloca ptr addrspace(3), align 4
  %block.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  %block = bitcast ptr addrspace(4) %.block_descriptor to ptr addrspace(4)
  store ptr addrspace(3) %p, ptr %p.addr, align 4
  store ptr addrspace(4) %block, ptr %block.addr, align 4
  ret void
}

define spir_kernel void @__device_side_enqueue_block_invoke_3_kernel(ptr addrspace(4) %0, ptr addrspace(3) %1) {
entry:
  call spir_func void @__device_side_enqueue_block_invoke_3(ptr addrspace(4) %0, ptr addrspace(3) %1)
  ret void
}

declare spir_func i32 @__enqueue_kernel_events_varargs(ptr, i32, ptr, i32, ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), i32, ptr)

define internal spir_func void @__device_side_enqueue_block_invoke_4(ptr addrspace(4) noundef %.block_descriptor, ptr addrspace(3) noundef %p1, ptr addrspace(3) noundef %p2, ptr addrspace(3) noundef %p3) {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  %p1.addr = alloca ptr addrspace(3), align 4
  %p2.addr = alloca ptr addrspace(3), align 4
  %p3.addr = alloca ptr addrspace(3), align 4
  %block.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  %block = bitcast ptr addrspace(4) %.block_descriptor to ptr addrspace(4)
  store ptr addrspace(3) %p1, ptr %p1.addr, align 4
  store ptr addrspace(3) %p2, ptr %p2.addr, align 4
  store ptr addrspace(3) %p3, ptr %p3.addr, align 4
  store ptr addrspace(4) %block, ptr %block.addr, align 4
  ret void
}

define spir_kernel void @__device_side_enqueue_block_invoke_4_kernel(ptr addrspace(4) %0, ptr addrspace(3) %1, ptr addrspace(3) %2, ptr addrspace(3) %3) {
entry:
  call spir_func void @__device_side_enqueue_block_invoke_4(ptr addrspace(4) %0, ptr addrspace(3) %1, ptr addrspace(3) %2, ptr addrspace(3) %3)
  ret void
}

declare spir_func i32 @__enqueue_kernel_varargs(ptr, i32, ptr, ptr addrspace(4), ptr addrspace(4), i32, ptr)

define internal spir_func void @__device_side_enqueue_block_invoke_5(ptr addrspace(4) noundef %.block_descriptor) {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  %block.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  %block = bitcast ptr addrspace(4) %.block_descriptor to ptr addrspace(4)
  store ptr addrspace(4) %block, ptr %block.addr, align 4
  %block.capture.addr = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 5
  %0 = load ptr addrspace(1), ptr addrspace(4) %block.capture.addr, align 4
  %block.capture.addr1 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 4
  %1 = load i32, ptr addrspace(4) %block.capture.addr1, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %0, i32 %1
  %2 = load i32, ptr addrspace(1) %arrayidx, align 4
  %block.capture.addr2 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 3
  %3 = load ptr addrspace(1), ptr addrspace(4) %block.capture.addr2, align 4
  %block.capture.addr3 = getelementptr inbounds <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, ptr addrspace(1) }>, ptr addrspace(4) %block, i32 0, i32 4
  %4 = load i32, ptr addrspace(4) %block.capture.addr3, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr addrspace(1) %3, i32 %4
  store i32 %2, ptr addrspace(1) %arrayidx4, align 4
  ret void
}

define spir_kernel void @__device_side_enqueue_block_invoke_5_kernel(ptr addrspace(4) %0) {
entry:
  call spir_func void @__device_side_enqueue_block_invoke_5(ptr addrspace(4) %0)
  ret void
}
