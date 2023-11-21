; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; This test checks that Invoke parameter of OpEnueueKernel instruction meet the
;; following specification requirements in case of enqueueing empty block:
;; "Invoke must be an OpFunction whose OpTypeFunction operand has:
;; - Result Type must be OpTypeVoid.
;; - The first parameter must have a type of OpTypePointer to an 8-bit OpTypeInt.
;; - An optional list of parameters, each of which must have a type of OpTypePointer to the Workgroup Storage Class.
;; ... "
;; __kernel void test_enqueue_empty() {
;;   enqueue_kernel(get_default_queue(),
;;                  CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
;;                  ndrange_1D(1),
;;                  0, NULL, NULL,
;;                  ^(){});
;; }

%struct.ndrange_t = type { i32, [3 x i64], [3 x i64], [3 x i64] }
%opencl.queue_t = type opaque
%opencl.clk_event_t = type opaque

@__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4

; CHECK-SPIRV: OpName %[[#Block:]] "__block_literal_global"
; CHECK-SPIRV: %[[#Void:]] = OpTypeVoid
; CHECK-SPIRV: %[[#Int8:]] = OpTypeInt 8
; CHECK-SPIRV: %[[#Int8PtrGen:]] = OpTypePointer Generic %[[#Int8]]
; CHECK-SPIRV: %[[#Int8Ptr:]] = OpTypePointer CrossWorkgroup %[[#Int8]]
; CHECK-SPIRV: %[[#Block]] = OpVariable %[[#]]

define spir_kernel void @test_enqueue_empty() {
entry:
  %tmp = alloca %struct.ndrange_t, align 8
  %call = call spir_func %opencl.queue_t* @_Z17get_default_queuev()
  call spir_func void @_Z10ndrange_1Dm(%struct.ndrange_t* sret(%struct.ndrange_t*) %tmp, i64 1)
  %0 = call i32 @__enqueue_kernel_basic_events(%opencl.queue_t* %call, i32 1, %struct.ndrange_t* %tmp, i32 0, %opencl.clk_event_t* addrspace(4)* null, %opencl.clk_event_t* addrspace(4)* null, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__test_enqueue_empty_block_invoke_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*))
  ret void
; CHECK-SPIRV: %[[#Int8PtrBlock:]] = OpBitcast %[[#Int8Ptr]] %[[#Block]]
; CHECK-SPIRV: %[[#Int8PtrGenBlock:]] = OpPtrCastToGeneric %[[#Int8PtrGen]] %[[#Int8PtrBlock]]
; CHECK-SPIRV: %[[#]] = OpEnqueueKernel %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#Invoke:]] %[[#Int8PtrGenBlock]] %[[#]] %[[#]]
}

declare spir_func %opencl.queue_t* @_Z17get_default_queuev()

declare spir_func void @_Z10ndrange_1Dm(%struct.ndrange_t* sret(%struct.ndrange_t*), i64)

define internal spir_func void @__test_enqueue_empty_block_invoke(i8 addrspace(4)* %.block_descriptor) {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 8
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 8
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  ret void
}

define internal spir_kernel void @__test_enqueue_empty_block_invoke_kernel(i8 addrspace(4)*) {
entry:
  call void @__test_enqueue_empty_block_invoke(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__enqueue_kernel_basic_events(%opencl.queue_t*, i32, %struct.ndrange_t*, i32, %opencl.clk_event_t* addrspace(4)*, %opencl.clk_event_t* addrspace(4)*, i8 addrspace(4)*, i8 addrspace(4)*)

; CHECK-SPIRV:      %[[#Invoke]] = OpFunction %[[#Void]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#]] = OpFunctionParameter %[[#Int8PtrGen]]
