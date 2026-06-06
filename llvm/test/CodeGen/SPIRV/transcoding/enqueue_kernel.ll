; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Kernel
; CHECK-DAG: %[[#typeInt64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#typeInt32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#typeInt8:]] = OpTypeInt 8 0

; CHECK-DAG: %[[#Num0i32:]] = OpConstantNull %[[#typeInt32]]
; CHECK-DAG: %[[#Num1i32:]] = OpConstant %[[#typeInt32]] 1 
; CHECK-DAG: %[[#Num2i32:]] = OpConstant %[[#typeInt32]] 2
; CHECK-DAG: %[[#Num3i32:]] = OpConstant %[[#typeInt32]] 3 
; CHECK-DAG: %[[#Num8i32:]] = OpConstant %[[#typeInt32]] 8 

; CHECK-DAG: %[[#Num16i32:]] = OpConstant %[[#typeInt32]] 16
; CHECK-DAG: %[[#Num29i32:]] = OpConstant %[[#typeInt32]] 29
; CHECK-DAG: %[[#Num36i32:]] = OpConstant %[[#typeInt32]] 36

; CHECK-DAG: %[[#Num1i64:]] = OpConstant %[[#typeInt64]] 1
; CHECK-DAG: %[[#Num2i64:]] = OpConstant %[[#typeInt64]] 2
; CHECK-DAG: %[[#Num4i64:]] = OpConstant %[[#typeInt64]] 4

; CHECK-DAG: %[[#Array3x64:]] = OpTypeArray %[[#typeInt64:]] %[[#Num3i32]]
; CHECK-DAG: %[[#TypeNDRangeStruct:]] = OpTypeStruct %[[#typeInt32]] %[[#Array3x64]] %[[#Array3x64]] %[[#Array3x64]]

; CHECK-DAG: %[[#pointerInt8:]] = OpTypePointer Generic %[[#typeInt8]]
; CHECK-DAG: %[[#nullArray3x64:]] = OpConstantNull %[[#Array3x64]]

; CHECK-DAG: %[[#typeEvent:]] = OpTypeDeviceEvent
; CHECK-DAG: %[[#typeEventPtr:]] = OpTypePointer Generic %[[#typeEvent]]
; CHECK-DAG: %[[#nullPtrEvent:]] = OpConstantNull %[[#typeEventPtr]]

; CHECK-DAG: %[[#typeVoid:]] = OpTypeVoid
; CHECK-DAG: %[[#typeWorkgroupPtrInt8:]] = OpTypePointer Workgroup %[[#typeInt8]]
; CHECK-DAG: %[[#typeFnVoidPtr:]] = OpTypeFunction %[[#typeVoid]] %[[#pointerInt8]]
; CHECK-DAG: %[[#typeFnVoidPtrLocal1:]] = OpTypeFunction %[[#typeVoid]] %[[#pointerInt8]] %[[#typeWorkgroupPtrInt8]]
; CHECK-DAG: %[[#typeFnVoidPtrLocal3:]] = OpTypeFunction %[[#typeVoid]] %[[#pointerInt8]] %[[#typeWorkgroupPtrInt8]] %[[#typeWorkgroupPtrInt8]] %[[#typeWorkgroupPtrInt8]]

; CHECK-DAG: OpName %[[#InvokeKernel1:]] "__device_side_enqueue_block_invoke_kernel"
; CHECK-DAG: OpName %[[#InvokeKernel2:]] "__device_side_enqueue_block_invoke_2_kernel"
; CHECK-DAG: OpName %[[#InvokeKernel3:]] "__device_side_enqueue_block_invoke_3_kernel"
; CHECK-DAG: OpName %[[#InvokeKernel4:]] "__device_side_enqueue_block_invoke_4_kernel"
; CHECK-DAG: OpName %[[#InvokeKernel5:]] "__device_side_enqueue_block_invoke_5_kernel"
; CHECK-DAG: OpName %[[#InvokeKernel6:]] "__device_side_enqueue_block_invoke_6_kernel"

; CHECK-LABEL: ; -- Begin function device_side_enqueue

; CHECK: %[[#NDRange3sret:]] = OpBuildNDRange %[[#TypeNDRangeStruct]] %[[#]] %[[#]] %[[#]]
; CHECK-NEXT: OpStore %[[#NDRange3:]] %[[#NDRange3sret]]

;; #define NULL ((void*)0)
;; kernel void device_side_enqueue(global int *a, global int *b, int i, char c0) {
;;     queue_t default_queue;
;;     unsigned flags = 0;
;;     ndrange_t ndrange;
;;     clk_event_t clk_event;
;;     clk_event_t event_wait_list;
;;     clk_event_t event_wait_list2[] = {clk_event};
;;
;;     const size_t gs[] = {1,2,4};
;;
;;     // enqueue empty kernel
; CHECK: %[[#]] = OpEnqueueKernel %[[#typeInt32]] %[[#default_queue:]] %[[#Num1i32]] %[[#NDRange3]] %[[#Num0i32]] %[[#nullPtrEvent]] %[[#nullPtrEvent]] %[[#InvokeKernel1]] %[[#]] %[[#Num16i32]] %[[#Num8i32]]
;;     enqueue_kernel(default_queue,
;;             CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
;;             ndrange_3D(gs),
;;             0, NULL, NULL,
;;             ^(){});
;;
;;     // no events, no var args
; CHECK: %[[#]] = OpEnqueueKernel %[[#typeInt32]] %[[#default_queue]] %[[#Num0i32]] %[[#]] %[[#Num0i32]] %[[#nullPtrEvent]] %[[#nullPtrEvent]] %[[#InvokeKernel2]] %[[#]] %[[#Num29i32]] %[[#Num8i32]]
;;     enqueue_kernel(default_queue, flags, ndrange,
;;             ^(void) {
;;             a[i] = c0;
;;             });
;;
;;     // event, no var args
; CHECK: %[[#event1:]] = OpPtrCastToGeneric %[[#typeEventPtr]] %[[#]]
; CHECK-NEXT: %[[#event2:]] = OpPtrCastToGeneric %[[#typeEventPtr]] %[[#]]
; CHECK: %[[#]] = OpEnqueueKernel %[[#typeInt32]] %[[#default_queue]] %[[#Num0i32]] %[[#]] %[[#Num2i32]] %[[#event1]] %[[#event2]] %[[#InvokeKernel3]] %[[#]] %[[#Num36i32]] %[[#Num8i32]]
;;     enqueue_kernel(default_queue, flags, ndrange, 2, &event_wait_list, &clk_event,
;;             ^(void) {
;;             a[i] = b[i];
;;             });
;;
;;     // events, var arg
; CHECK: %[[#]] = OpEnqueueKernel %[[#typeInt32]] %[[#default_queue]] %[[#Num0i32]] %[[#]] %[[#Num2i32]] %[[#event_wait_list2:]] %[[#event2]] %[[#InvokeKernel4]] %[[#]] %[[#Num16i32]] %[[#Num8i32]] %[[#]]
;;     char c;
;;     enqueue_kernel(default_queue, flags, ndrange, 2, event_wait_list2, &clk_event,
;;             ^(local void *p) {
;;             return;
;;             },
;;             c);
;;
;;     // no events, three var args
; CHECK: %[[#]] = OpEnqueueKernel %[[#typeInt32]] %[[#default_queue]] %[[#Num0i32]] %[[#]] %[[#Num0i32]] %[[#nullPtrEvent]] %[[#nullPtrEvent]] %[[#InvokeKernel5]] %[[#]] %[[#Num16i32]] %[[#Num8i32]] %[[#]] %[[#]] %[[#]]
;;     enqueue_kernel(default_queue, flags, ndrange,
;;             ^(local void *p1, local void *p2, local void *p3) {
;;             return;
;;             },
;;             101, 102, 104);
;;
;;     // null event, no var args
; CHECK: %[[#]] = OpEnqueueKernel %[[#typeInt32]] %[[#default_queue]] %[[#Num0i32]] %[[#]] %[[#Num0i32]] %[[#nullPtrEvent]] %[[#event2]] %[[#InvokeKernel6]] %[[#]] %[[#Num36i32]] %[[#Num8i32]]
;;     enqueue_kernel(default_queue, flags, ndrange, 0, NULL, &clk_event,
;;             ^(void) {
;;             a[i] = b[i];
;;             });
;; }

; CHECK-DAG: %[[#InvokeKernel1]] = OpFunction %[[#typeVoid]] {{Pure|None}} %[[#typeFnVoidPtr]]
; CHECK-DAG: %[[#InvokeKernel2]] = OpFunction %[[#typeVoid]] {{Pure|None}} %[[#typeFnVoidPtr]]
; CHECK-DAG: %[[#InvokeKernel3]] = OpFunction %[[#typeVoid]] {{Pure|None}} %[[#typeFnVoidPtr]]
; CHECK-DAG: %[[#InvokeKernel4]] = OpFunction %[[#typeVoid]] {{Pure|None}} %[[#typeFnVoidPtrLocal1]]
; CHECK-DAG: %[[#InvokeKernel5]] = OpFunction %[[#typeVoid]] {{Pure|None}} %[[#typeFnVoidPtrLocal3]]
; CHECK-DAG: %[[#InvokeKernel6]] = OpFunction %[[#typeVoid]] {{Pure|None}} %[[#typeFnVoidPtr]]
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

%struct.ndrange_t = type { i32, [3 x i64], [3 x i64], [3 x i64] }

@__const.device_side_enqueue.gs = private unnamed_addr addrspace(2) constant [3 x i64] [i64 1, i64 2, i64 4], align 8
@__block_literal_global = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 16, i32 8, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke to ptr addrspace(4)) }, align 8
@__block_literal_global.1 = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 16, i32 8, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_4 to ptr addrspace(4)) }, align 8
@__block_literal_global.2 = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 16, i32 8, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_5 to ptr addrspace(4)) }, align 8

define spir_kernel void @device_side_enqueue(ptr addrspace(1) align 4 %a, ptr addrspace(1) align 4 %b, i32 %i, i8 %c0, target("spirv.Queue") %default_queue) {
entry:
  %clk_event.i = alloca target("spirv.DeviceEvent"), align 8
  %event_wait_list.i = alloca target("spirv.DeviceEvent"), align 8
  %event_wait_list2.i = alloca [1 x target("spirv.DeviceEvent")], align 8
  %gs.i = alloca [3 x i64], align 8
  %tmp.i = alloca %struct.ndrange_t, align 8
  %tmp1.i = alloca %struct.ndrange_t, align 8
  %block.i = alloca <{ i32, i32, ptr addrspace(4), ptr addrspace(1), i32, i8 }>, align 8
  %tmp4.i = alloca %struct.ndrange_t, align 8
  %block5.i = alloca <{ i32, i32, ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32 }>, align 8
  %tmp12.i = alloca %struct.ndrange_t, align 8
  %block_sizes.i = alloca [1 x i64], align 8
  %tmp14.i = alloca %struct.ndrange_t, align 8
  %block_sizes15.i = alloca [3 x i64], align 8
  %tmp16.i = alloca %struct.ndrange_t, align 8
  %block17.i = alloca <{ i32, i32, ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32 }>, align 8
  call void @llvm.memcpy.p0.p2.i64(ptr align 8 %gs.i, ptr addrspace(2) align 8 @__const.device_side_enqueue.gs, i64 24, i1 false)
  call spir_func void @_Z10ndrange_3DPKm(ptr sret(%struct.ndrange_t) align 8 %tmp.i, ptr %gs.i)
  %0 = call spir_func i32 @__enqueue_kernel_basic_events(target("spirv.Queue") %default_queue, i32 1, ptr %tmp.i, i32 0, ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global to ptr addrspace(4)))
  store i32 29, ptr %block.i, align 8
  %block.align.i = getelementptr inbounds i8, ptr %block.i, i64 4
  store i32 8, ptr %block.align.i, align 4
  %block.invoke.i = getelementptr inbounds i8, ptr %block.i, i64 8
  store ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_2 to ptr addrspace(4)), ptr %block.invoke.i, align 8
  %block.captured.i = getelementptr inbounds i8, ptr %block.i, i64 16
  store ptr addrspace(1) %a, ptr %block.captured.i, align 8
  %block.captured2.i = getelementptr inbounds i8, ptr %block.i, i64 24
  store i32 %i, ptr %block.captured2.i, align 8
  %block.captured3.i = getelementptr inbounds i8, ptr %block.i, i64 28
  store i8 %c0, ptr %block.captured3.i, align 4
  %1 = addrspacecast ptr %block.i to ptr addrspace(4)
  %2 = call spir_func i32 @__enqueue_kernel_basic(target("spirv.Queue") %default_queue, i32 0, ptr %tmp1.i, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_2_kernel to ptr addrspace(4)), ptr addrspace(4) %1)
  %3 = addrspacecast ptr %event_wait_list.i to ptr addrspace(4)
  %4 = addrspacecast ptr %clk_event.i to ptr addrspace(4)
  store i32 36, ptr %block5.i, align 8
  %block.align7.i = getelementptr inbounds i8, ptr %block5.i, i64 4
  store i32 8, ptr %block.align7.i, align 4
  %block.invoke8.i = getelementptr inbounds i8, ptr %block5.i, i64 8
  store ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_3 to ptr addrspace(4)), ptr %block.invoke8.i, align 8
  %block.captured9.i = getelementptr inbounds i8, ptr %block5.i, i64 16
  store ptr addrspace(1) %a, ptr %block.captured9.i, align 8
  %block.captured10.i = getelementptr inbounds i8, ptr %block5.i, i64 32
  store i32 %i, ptr %block.captured10.i, align 8
  %block.captured11.i = getelementptr inbounds i8, ptr %block5.i, i64 24
  store ptr addrspace(1) %b, ptr %block.captured11.i, align 8
  %5 = addrspacecast ptr %block5.i to ptr addrspace(4)
  %6 = call spir_func i32 @__enqueue_kernel_basic_events(target("spirv.Queue") %default_queue, i32 0, ptr %tmp4.i, i32 2, ptr addrspace(4) %3, ptr addrspace(4) %4, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_3_kernel to ptr addrspace(4)), ptr addrspace(4) %5)
  %7 = addrspacecast ptr %event_wait_list2.i to ptr addrspace(4)
  store i64 0, ptr %block_sizes.i, align 8
  %8 = call spir_func i32 @__enqueue_kernel_events_varargs(target("spirv.Queue") %default_queue, i32 0, ptr %tmp12.i, i32 2, ptr addrspace(4) %7, ptr addrspace(4) %4, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_4_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global.1 to ptr addrspace(4)), i32 1, ptr %block_sizes.i)
  store i64 101, ptr %block_sizes15.i, align 8
  %9 = getelementptr inbounds i8, ptr %block_sizes15.i, i64 8
  store i64 102, ptr %9, align 8
  %10 = getelementptr inbounds i8, ptr %block_sizes15.i, i64 16
  store i64 104, ptr %10, align 8
  %11 = call spir_func i32 @__enqueue_kernel_varargs(target("spirv.Queue") %default_queue, i32 0, ptr %tmp14.i, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_5_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global.2 to ptr addrspace(4)), i32 3, ptr %block_sizes15.i)
  store i32 36, ptr %block17.i, align 8
  %block.align19.i = getelementptr inbounds i8, ptr %block17.i, i64 4
  store i32 8, ptr %block.align19.i, align 4
  %block.invoke20.i = getelementptr inbounds i8, ptr %block17.i, i64 8
  store ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_6 to ptr addrspace(4)), ptr %block.invoke20.i, align 8
  %block.captured21.i = getelementptr inbounds i8, ptr %block17.i, i64 16
  store ptr addrspace(1) %a, ptr %block.captured21.i, align 8
  %block.captured22.i = getelementptr inbounds i8, ptr %block17.i, i64 32
  store i32 %i, ptr %block.captured22.i, align 8
  %block.captured23.i = getelementptr inbounds i8, ptr %block17.i, i64 24
  store ptr addrspace(1) %b, ptr %block.captured23.i, align 8
  %12 = addrspacecast ptr %block17.i to ptr addrspace(4)
  %13 = call spir_func i32 @__enqueue_kernel_basic_events(target("spirv.Queue") %default_queue, i32 0, ptr %tmp16.i, i32 0, ptr addrspace(4) null, ptr addrspace(4) %4, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_6_kernel to ptr addrspace(4)), ptr addrspace(4) %12)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr)

declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1 immarg)

declare spir_func void @_Z10ndrange_3DPKm(ptr sret(%struct.ndrange_t) align 8, ptr)

define internal spir_func void @__device_side_enqueue_block_invoke(ptr addrspace(4) %.block_descriptor) {
entry:
  ret void
}

define internal spir_kernel void @__device_side_enqueue_block_invoke_kernel(ptr addrspace(4) %0) {
entry:
  ret void
}

declare spir_func i32 @__enqueue_kernel_basic_events(target("spirv.Queue"), i32, ptr, i32, ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4))

define internal spir_func void @__device_side_enqueue_block_invoke_2(ptr addrspace(4) %.block_descriptor) {
  ret void
}

define internal spir_kernel void @__device_side_enqueue_block_invoke_2_kernel(ptr addrspace(4) %0) {
  ret void
}

declare spir_func i32 @__enqueue_kernel_basic(target("spirv.Queue"), i32, ptr, ptr addrspace(4), ptr addrspace(4))

define internal spir_func void @__device_side_enqueue_block_invoke_3(ptr addrspace(4) %.block_descriptor) {
  ret void
}

define internal spir_kernel void @__device_side_enqueue_block_invoke_3_kernel(ptr addrspace(4) %0) {
  ret void
}

define internal spir_func void @__device_side_enqueue_block_invoke_4(ptr addrspace(4) %.block_descriptor, ptr addrspace(3) %p) {
entry:
  ret void
}

define internal spir_kernel void @__device_side_enqueue_block_invoke_4_kernel(ptr addrspace(4) %0, ptr addrspace(3) %1) {
entry:
  ret void
}

declare spir_func i32 @__enqueue_kernel_events_varargs(target("spirv.Queue"), i32, ptr, i32, ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), i32, ptr)

declare void @llvm.lifetime.end.p0(ptr)

define internal spir_func void @__device_side_enqueue_block_invoke_5(ptr addrspace(4) %.block_descriptor, ptr addrspace(3) %p1, ptr addrspace(3) %p2, ptr addrspace(3) %p3) {
entry:
  ret void
}

define internal spir_kernel void @__device_side_enqueue_block_invoke_5_kernel(ptr addrspace(4) %0, ptr addrspace(3) %1, ptr addrspace(3) %2, ptr addrspace(3) %3) {
entry:
  ret void
}

declare spir_func i32 @__enqueue_kernel_varargs(target("spirv.Queue"), i32, ptr, ptr addrspace(4), ptr addrspace(4), i32, ptr)

define internal spir_func void @__device_side_enqueue_block_invoke_6(ptr addrspace(4) %.block_descriptor) {
  ret void
}

define internal spir_kernel void @__device_side_enqueue_block_invoke_6_kernel(ptr addrspace(4) %0) {
  ret void
}


!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
