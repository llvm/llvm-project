; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *

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
  %call = call spir_func ptr @_Z17get_default_queuev()
  call spir_func void @_Z10ndrange_1Dm(ptr sret(ptr) %tmp, i64 1)
  %0 = call i32 @__enqueue_kernel_basic_events(ptr %call, i32 1, ptr %tmp, i32 0, ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @__test_enqueue_empty_block_invoke_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global to ptr addrspace(4)))
  ret void
; CHECK-SPIRV: %[[#Int8PtrBlock:]] = OpBitcast %[[#Int8Ptr]] %[[#Block]]
; CHECK-SPIRV: %[[#Int8PtrGenBlock:]] = OpPtrCastToGeneric %[[#Int8PtrGen]] %[[#Int8PtrBlock]]
; CHECK-SPIRV: %[[#]] = OpEnqueueKernel %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#Invoke:]] %[[#Int8PtrGenBlock]] %[[#]] %[[#]]
}

declare spir_func ptr @_Z17get_default_queuev()

declare spir_func void @_Z10ndrange_1Dm(ptr sret(ptr), i64)

define internal spir_func void @__test_enqueue_empty_block_invoke(ptr addrspace(4) %.block_descriptor) {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 8
  %block = bitcast ptr addrspace(4) %.block_descriptor to ptr addrspace(4)
  ret void
}

define internal spir_kernel void @__test_enqueue_empty_block_invoke_kernel(ptr addrspace(4)) {
entry:
  call void @__test_enqueue_empty_block_invoke(ptr addrspace(4) %0)
  ret void
}

declare i32 @__enqueue_kernel_basic_events(ptr, i32, ptr, i32, ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4))

; CHECK-SPIRV:      %[[#Invoke]] = OpFunction %[[#Void]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#]] = OpFunctionParameter %[[#Int8PtrGen]]
