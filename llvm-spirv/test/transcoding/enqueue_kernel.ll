; Source
; typedef struct {int a;} ndrange_t;
;
; kernel void device_side_enqueue(global int *a, global int *b, int i, char c0) {
;   queue_t default_queue;
;   unsigned flags = 0;
;   ndrange_t ndrange;
;   clk_event_t clk_event;
;   clk_event_t event_wait_list;
;   clk_event_t event_wait_list2[] = {clk_event};
;
;   // Emits block literal on stack and block kernel.
;   enqueue_kernel(default_queue, flags, ndrange,
;                  ^(void) {
;                    a[i] = c0;
;                  });
;
;   // Emits block literal on stack and block kernel.
;   enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
;                  ^(void) {
;                    a[i] = b[i];
;                  });
;
;   char c;
;   // Emits global block literal and block kernel.
;   enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
;                  ^(local void *p) {
;                    return;
;                  },
;                  c);
;
;   // Emits global block literal and block kernel.
;   enqueue_kernel(default_queue, flags, ndrange,
;                  ^(local void *p1, local void *p2, local void *p3) {
;                    return;
;                  },
;                  1, 2, 4);
; }
;
; Compilation command:
; clang -cc1 -triple spir-unknown-unknown -O0 -cl-std=CL2.0 -emit-llvm enqueue_kernel.cl

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; ModuleID = 'enqueue_kernel.cl'
source_filename = "enqueue_kernel.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.queue_t = type opaque
%struct.ndrange_t = type { i32 }
%opencl.clk_event_t = type opaque

; CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer1:[0-9]+]] "__device_side_enqueue_block_invoke_kernel"
; CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer2:[0-9]+]] "__device_side_enqueue_block_invoke_2_kernel"
; CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer3:[0-9]+]] "__device_side_enqueue_block_invoke_3_kernel"
; CHECK-SPIRV: EntryPoint {{[0-9]+}} [[BlockKer4:[0-9]+]] "__device_side_enqueue_block_invoke_4_kernel"
; CHECK-SPIRV: Name [[BlockGlb1:[0-9]+]] "__block_literal_global"
; CHECK-SPIRV: Name [[BlockGlb2:[0-9]+]] "__block_literal_global.1"

; CHECK-SPIRV: TypeInt [[Int32Ty:[0-9]+]] 32
; CHECK-SPIRV: TypeInt [[Int8Ty:[0-9]+]] 8
; CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt8:[0-9]+]] 8
; CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt0:[0-9]+]] 0
; CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt17:[0-9]+]] 17
; CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt2:[0-9]+]] 2
; CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt20:[0-9]+]] 20
; CHECK-SPIRV: TypeVoid [[VoidTy:[0-9]+]]

; CHECK-SPIRV: TypePointer {{[0-9]+}} 7 {{[0-9]+}}
; CHECK-SPIRV: TypePointer [[Int32LocPtrTy:[0-9]+]] 7 [[Int32Ty]]
; CHECK-SPIRV: TypeDeviceEvent [[EventTy:[0-9]+]]
; CHECK-SPIRV: TypePointer [[Int8PtrGenTy:[0-9]+]] 8 [[Int8Ty]]
; CHECK-SPIRV: TypePointer [[EventPtrTy:[0-9]+]] 8 [[EventTy]]
; CHECK-SPIRV: TypeFunction [[BlockTy1:[0-9]+]] [[VoidTy]] [[Int8PtrGenTy]]
; CHECK-SPIRV: TypeFunction [[BlockTy2:[0-9]+]] [[VoidTy]] [[Int8PtrGenTy]]
; CHECK-SPIRV: TypeFunction [[BlockTy3:[0-9]+]] [[VoidTy]] [[Int8PtrGenTy]]
; CHECK-SPIRV: ConstantNull [[EventPtrTy]] [[EventNull:[0-9]+]]

; CHECK-LLVM: [[BlockTy1:%[0-9]+]] = type { i32, i32 }
; CHECK-LLVM: [[BlockTy2:%[0-9]+]] = type <{ i32, i32, i32 addrspace(1)*, i32, i8 }>
; CHECK-LLVM: [[BlockTy3:%[0-9]+]] = type <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>
; CHECK-LLVM: [[BlockTy4:%[0-9]+]] = type <{ i32, i32 }>

; CHECK-LLVM: @__block_literal_global = internal addrspace(1) constant [[BlockTy1]] { i32 8, i32 4 }, align 4
; CHECK-LLVM: @__block_literal_global.1 = internal addrspace(1) constant [[BlockTy1]] { i32 8, i32 4 }, align 4

@__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4
@__block_literal_global.1 = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @device_side_enqueue(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %i, i8 signext %c0) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %default_queue = alloca %opencl.queue_t*, align 4
  %flags = alloca i32, align 4
  %ndrange = alloca %struct.ndrange_t, align 4
  %clk_event = alloca %opencl.clk_event_t*, align 4
  %event_wait_list = alloca %opencl.clk_event_t*, align 4
  %event_wait_list2 = alloca [1 x %opencl.clk_event_t*], align 4
  %block = alloca <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, align 4
  %block3 = alloca <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, align 4
  %c = alloca i8, align 1
  store i32 0, i32* %flags, align 4
  %arrayinit.begin = getelementptr inbounds [1 x %opencl.clk_event_t*], [1 x %opencl.clk_event_t*]* %event_wait_list2, i32 0, i32 0
  %0 = load %opencl.clk_event_t*, %opencl.clk_event_t** %clk_event, align 4
  store %opencl.clk_event_t* %0, %opencl.clk_event_t** %arrayinit.begin, align 4
  %1 = load %opencl.queue_t*, %opencl.queue_t** %default_queue, align 4
  %2 = load i32, i32* %flags, align 4
  %block.size = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }>* %block, i32 0, i32 0
  store i32 17, i32* %block.size, align 4
  %block.align = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }>* %block, i32 0, i32 1
  store i32 4, i32* %block.align, align 4
  %block.captured = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }>* %block, i32 0, i32 2
  store i32 addrspace(1)* %a, i32 addrspace(1)** %block.captured, align 4
  %block.captured1 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }>* %block, i32 0, i32 3
  store i32 %i, i32* %block.captured1, align 4
  %block.captured2 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }>* %block, i32 0, i32 4
  store i8 %c0, i8* %block.captured2, align 4
  %3 = bitcast <{ i32, i32, i32 addrspace(1)*, i32, i8 }>* %block to void ()*
  %4 = addrspacecast void ()* %3 to i8 addrspace(4)*

; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit1:[0-9]+]]
; CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
;                            [[ConstInt0]] [[EventNull]] [[EventNull]]
;                            [[BlockKer1]] [[BlockLit1]] [[ConstInt17]] [[ConstInt8]]

; CHECK-LLVM: [[Block2:%[0-9]+]] = addrspacecast [[BlockTy2]]* %block to i8 addrspace(4)*
; CHECK-LLVM: [[BlockInv2:%[0-9]+]] = addrspacecast void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_kernel to i8 addrspace(4)*
; CHECK-LLVM: call i32 @__enqueue_kernel_basic_events(%opencl.queue_t* {{.*}}, i32 {{.*}}, %struct.ndrange_t* {{.*}}, i32 0, %opencl.clk_event_t* addrspace(4)* null, %opencl.clk_event_t* addrspace(4)* null, i8 addrspace(4)* [[BlockInv2]], i8 addrspace(4)* [[Block2]])

  %5 = call i32 @__enqueue_kernel_basic(%opencl.queue_t* %1, i32 %2, %struct.ndrange_t* byval %ndrange, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* %4)
  %6 = addrspacecast %opencl.clk_event_t** %event_wait_list to %opencl.clk_event_t* addrspace(4)*
  %7 = addrspacecast %opencl.clk_event_t** %clk_event to %opencl.clk_event_t* addrspace(4)*
  %block.size5 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>* %block3, i32 0, i32 0
  store i32 20, i32* %block.size5, align 4
  %block.align6 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>* %block3, i32 0, i32 1
  store i32 4, i32* %block.align6, align 4
  %block.captured7 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>* %block3, i32 0, i32 2
  store i32 addrspace(1)* %a, i32 addrspace(1)** %block.captured7, align 4
  %block.captured8 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>* %block3, i32 0, i32 3
  store i32 %i, i32* %block.captured8, align 4
  %block.captured9 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>* %block3, i32 0, i32 4
  store i32 addrspace(1)* %b, i32 addrspace(1)** %block.captured9, align 4
  %8 = bitcast <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>* %block3 to void ()*
  %9 = addrspacecast void ()* %8 to i8 addrspace(4)*

; CHECK-SPIRV: PtrCastToGeneric [[EventPtrTy]] [[Event1:[0-9]+]]
; CHECK-SPIRV: PtrCastToGeneric [[EventPtrTy]] [[Event2:[0-9]+]]

; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit2:[0-9]+]]
; CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
;                            [[ConstInt2]] [[Event1]] [[Event2]]
;                            [[BlockKer2]] [[BlockLit2]] [[ConstInt20]] [[ConstInt8]]

; CHECK-LLVM: [[Block3:%[0-9]+]] = addrspacecast [[BlockTy3]]* %block3 to i8 addrspace(4)*
; CHECK-LLVM: [[BlockInv3:%[0-9]+]] = addrspacecast void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_2_kernel to i8 addrspace(4)*
; CHECK-LLVM: call i32 @__enqueue_kernel_basic_events(%opencl.queue_t* {{.*}}, i32 {{.*}}, %struct.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t* addrspace(4)* {{.*}}, %opencl.clk_event_t* addrspace(4)* {{.*}}, i8 addrspace(4)* [[BlockInv3]], i8 addrspace(4)* [[Block3]])

  %10 = call i32 @__enqueue_kernel_basic_events(%opencl.queue_t* %1, i32 %2, %struct.ndrange_t* %ndrange, i32 2, %opencl.clk_event_t* addrspace(4)* %6, %opencl.clk_event_t* addrspace(4)* %7, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_2_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* %9)
  %11 = alloca [1 x i32]
  %12 = getelementptr [1 x i32], [1 x i32]* %11, i32 0, i32 0
  %13 = load i8, i8* %c, align 1
  %14 = zext i8 %13 to i32
  store i32 %14, i32* %12, align 4

; CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf31:[0-9]+]]
; CHECK-SPIRV: Bitcast {{[0-9]+}} [[BlockLit3Tmp:[0-9]+]] [[BlockGlb1:[0-9]+]]
; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit3:[0-9]+]] [[BlockLit3Tmp]]
; CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
;                            [[ConstInt2]] [[Event1]] [[Event2]]
;                            [[BlockKer3]] [[BlockLit3]] [[ConstInt8]] [[ConstInt8]]
;                            [[LocalBuf31]]

; CHECK-LLVM: [[Block0Tmp:%[0-9]+]] = bitcast [[BlockTy1]] addrspace(1)* @__block_literal_global to i8 addrspace(1)*
; CHECK-LLVM: [[Block0:%[0-9]+]] = addrspacecast i8 addrspace(1)* [[Block0Tmp]] to i8 addrspace(4)*
; CHECK-LLVM: [[BlockInv0:%[0-9]+]] = addrspacecast void (i8 addrspace(4)*, i8 addrspace(3)*)* @__device_side_enqueue_block_invoke_3_kernel to i8 addrspace(4)*
; CHECK-LLVM: call i32 @__enqueue_kernel_events_varargs(%opencl.queue_t* {{.*}}, i32 {{.*}}, %struct.ndrange_t* {{.*}}, i32 2, %opencl.clk_event_t* addrspace(4)* {{.*}}, %opencl.clk_event_t* addrspace(4)* {{.*}}, i8 addrspace(4)* [[BlockInv0]], i8 addrspace(4)* [[Block0]], i32 1, i32* {{.*}})

  %15 = call i32 @__enqueue_kernel_events_varargs(%opencl.queue_t* %1, i32 %2, %struct.ndrange_t* %ndrange, i32 2, %opencl.clk_event_t* addrspace(4)* %6, %opencl.clk_event_t* addrspace(4)* %7, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* @__device_side_enqueue_block_invoke_3_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*), i32 1, i32* %12)
  %16 = alloca [3 x i32]
  %17 = getelementptr [3 x i32], [3 x i32]* %16, i32 0, i32 0
  store i32 1, i32* %17, align 4
  %18 = getelementptr [3 x i32], [3 x i32]* %16, i32 0, i32 1
  store i32 2, i32* %18, align 4
  %19 = getelementptr [3 x i32], [3 x i32]* %16, i32 0, i32 2
  store i32 4, i32* %19, align 4

; CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf41:[0-9]+]]
; CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf42:[0-9]+]]
; CHECK-SPIRV: PtrAccessChain [[Int32LocPtrTy]] [[LocalBuf43:[0-9]+]]
; CHECK-SPIRV: Bitcast {{[0-9]+}} [[BlockLit4Tmp:[0-9]+]] [[BlockGlb2:[0-9]+]]
; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit4:[0-9]+]] [[BlockLit4Tmp]]
; CHECK-SPIRV: EnqueueKernel [[Int32Ty]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
;                            [[ConstInt0]] [[EventNull]] [[EventNull]]
;                            [[BlockKer4]] [[BlockLit4]] [[ConstInt8]] [[ConstInt8]]
;                            [[LocalBuf41]] [[LocalBuf42]] [[LocalBuf43]]

; CHECK-LLVM: [[Block1Tmp:%[0-9]+]] = bitcast [[BlockTy1]] addrspace(1)* @__block_literal_global.1 to i8 addrspace(1)*
; CHECK-LLVM: [[Block1:%[0-9]+]] = addrspacecast i8 addrspace(1)* [[Block1Tmp]] to i8 addrspace(4)*
; CHECK-LLVM: [[BlockInv1:%[0-9]+]] = addrspacecast void (i8 addrspace(4)*, i8 addrspace(3)*, i8 addrspace(3)*, i8 addrspace(3)*)* @__device_side_enqueue_block_invoke_4_kernel to i8 addrspace(4)*
; CHECK-LLVM: call i32 @__enqueue_kernel_events_varargs(%opencl.queue_t* {{.*}}, i32 {{.*}}, %struct.ndrange_t* {{.*}}, i32 0, %opencl.clk_event_t* addrspace(4)* null, %opencl.clk_event_t* addrspace(4)* null, i8 addrspace(4)* [[BlockInv1]], i8 addrspace(4)* [[Block1]], i32 3, i32* {{.*}})

  %20 = call i32 @__enqueue_kernel_varargs(%opencl.queue_t* %1, i32 %2, %struct.ndrange_t* %ndrange, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*, i8 addrspace(3)*, i8 addrspace(3)*)* @__device_side_enqueue_block_invoke_4_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global.1 to i8 addrspace(1)*) to i8 addrspace(4)*), i32 3, i32* %17)
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke(i8 addrspace(4)* %.block_descriptor) #2 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32, i32 addrspace(1)*, i32, i8 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i32 addrspace(1)*, i32, i8 }> addrspace(4)*
  store <{ i32, i32, i32 addrspace(1)*, i32, i8 }> addrspace(4)* %block, <{ i32, i32, i32 addrspace(1)*, i32, i8 }> addrspace(4)** %block.addr, align 4
  %block.capture.addr = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }> addrspace(4)* %block, i32 0, i32 4
  %0 = load i8, i8 addrspace(4)* %block.capture.addr, align 4
  %conv = sext i8 %0 to i32
  %block.capture.addr1 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }> addrspace(4)* %block, i32 0, i32 2
  %1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %block.capture.addr1, align 4
  %block.capture.addr2 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i8 }>, <{ i32, i32, i32 addrspace(1)*, i32, i8 }> addrspace(4)* %block, i32 0, i32 3
  %2 = load i32, i32 addrspace(4)* %block.capture.addr2, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %1, i32 %2
  store i32 %conv, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_kernel(i8 addrspace(4)*) #3 {
entry:
  call void @__device_side_enqueue_block_invoke(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__enqueue_kernel_basic(%opencl.queue_t*, i32, %struct.ndrange_t*, i8 addrspace(4)*, i8 addrspace(4)*)

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke_2(i8 addrspace(4)* %.block_descriptor) #2 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)*
  store <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)* %block, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)** %block.addr, align 4
  %block.capture.addr = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)* %block, i32 0, i32 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %block.capture.addr, align 4
  %block.capture.addr1 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)* %block, i32 0, i32 3
  %1 = load i32, i32 addrspace(4)* %block.capture.addr1, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i32 %1
  %2 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %block.capture.addr2 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)* %block, i32 0, i32 2
  %3 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %block.capture.addr2, align 4
  %block.capture.addr3 = getelementptr inbounds <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }>, <{ i32, i32, i32 addrspace(1)*, i32, i32 addrspace(1)* }> addrspace(4)* %block, i32 0, i32 3
  %4 = load i32, i32 addrspace(4)* %block.capture.addr3, align 4
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %3, i32 %4
  store i32 %2, i32 addrspace(1)* %arrayidx4, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_2_kernel(i8 addrspace(4)*) #3 {
entry:
  call void @__device_side_enqueue_block_invoke_2(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__enqueue_kernel_basic_events(%opencl.queue_t*, i32, %struct.ndrange_t*, i32, %opencl.clk_event_t* addrspace(4)*, %opencl.clk_event_t* addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*)

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke_3(i8 addrspace(4)* %.block_descriptor, i8 addrspace(3)* %p) #2 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %p.addr = alloca i8 addrspace(3)*, align 4
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store i8 addrspace(3)* %p, i8 addrspace(3)** %p.addr, align 4
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_3_kernel(i8 addrspace(4)*, i8 addrspace(3)*) #3 {
entry:
  call void @__device_side_enqueue_block_invoke_3(i8 addrspace(4)* %0, i8 addrspace(3)* %1)
  ret void
}

declare i32 @__enqueue_kernel_events_varargs(%opencl.queue_t*, i32, %struct.ndrange_t*, i32, %opencl.clk_event_t* addrspace(4)*, %opencl.clk_event_t* addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*, i32, i32*)

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke_4(i8 addrspace(4)* %.block_descriptor, i8 addrspace(3)* %p1, i8 addrspace(3)* %p2, i8 addrspace(3)* %p3) #2 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %p1.addr = alloca i8 addrspace(3)*, align 4
  %p2.addr = alloca i8 addrspace(3)*, align 4
  %p3.addr = alloca i8 addrspace(3)*, align 4
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store i8 addrspace(3)* %p1, i8 addrspace(3)** %p1.addr, align 4
  store i8 addrspace(3)* %p2, i8 addrspace(3)** %p2.addr, align 4
  store i8 addrspace(3)* %p3, i8 addrspace(3)** %p3.addr, align 4
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_4_kernel(i8 addrspace(4)*, i8 addrspace(3)*, i8 addrspace(3)*, i8 addrspace(3)*) #3 {
entry:
  call void @__device_side_enqueue_block_invoke_4(i8 addrspace(4)* %0, i8 addrspace(3)* %1, i8 addrspace(3)* %2, i8 addrspace(3)* %3)
  ret void
}

declare i32 @__enqueue_kernel_varargs(%opencl.queue_t*, i32, %struct.ndrange_t*, i8 addrspace(4)*, i8 addrspace(4)*, i32, i32*)

; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer1]] 0 [[BlockTy1]]
; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer2]] 0 [[BlockTy1]]
; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer3]] 0 [[BlockTy2]]
; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer4]] 0 [[BlockTy3]]

; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_kernel(i8 addrspace(4)*)
; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_2_kernel(i8 addrspace(4)*)
; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_3_kernel(i8 addrspace(4)*, i8 addrspace(3)*)
; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_4_kernel(i8 addrspace(4)*, i8 addrspace(3)*, i8 addrspace(3)*, i8 addrspace(3)*)

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"clang version 7.0.0"}
!4 = !{i32 1, i32 1, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none"}
!6 = !{!"int*", !"int*", !"int", !"char"}
!7 = !{!"", !"", !"", !""}
!8 = !{i1 false, i1 false, i1 false, i1 false}
!9 = !{i32 0, i32 0, i32 0, i32 0}
