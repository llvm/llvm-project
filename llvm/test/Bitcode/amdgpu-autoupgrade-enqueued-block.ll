; RUN: llvm-as < %s | llvm-dis | FileCheck %s

%struct.ndrange_t = type { i32 }
%opencl.queue_t = type opaque

; CHECK: %block.runtime.handle.t = type { ptr, i32, i32 }
; CHECK: %block.runtime.handle.t.0 = type { ptr, i32, i32 }
; CHECK: %block.runtime.handle.t.1 = type { ptr, i32, i32 }
; CHECK: %block.runtime.handle.t.2 = type { ptr, i32, i32 }
; CHECK: %block.runtime.handle.t.3 = type { ptr, i32, i32 }
; CHECK: %block.runtime.handle.t.4 = type { ptr, i32, i32 }


; CHECK: @kernel_address_user = global [1 x ptr] [ptr @block_has_used_kernel_address]
; CHECK: @__test_block_invoke_kernel.runtime.handle = internal externally_initialized constant %block.runtime.handle.t zeroinitializer, section ".amdgpu.kernel.runtime.handle"
; CHECK: @__test_block_invoke_2_kernel.runtime.handle = internal externally_initialized constant %block.runtime.handle.t.0 zeroinitializer, section ".amdgpu.kernel.runtime.handle"
; CHECK: @block_has_used_kernel_address.runtime.handle = internal externally_initialized constant %block.runtime.handle.t.1 zeroinitializer, section ".amdgpu.kernel.runtime.handle"
; CHECK: @.runtime.handle = internal externally_initialized constant %block.runtime.handle.t.2 zeroinitializer, section ".amdgpu.kernel.runtime.handle"
; CHECK: @.runtime.handle.1 = internal externally_initialized constant %block.runtime.handle.t.3 zeroinitializer, section ".amdgpu.kernel.runtime.handle"
; CHECK: @kernel_linkonce_odr_block.runtime.handle = linkonce_odr externally_initialized constant %block.runtime.handle.t.4 zeroinitializer, section ".amdgpu.kernel.runtime.handle"
; CHECK: @llvm.used = appending global [12 x ptr] [ptr @__test_block_invoke_kernel, ptr @__test_block_invoke_kernel.runtime.handle, ptr @__test_block_invoke_2_kernel, ptr @__test_block_invoke_2_kernel.runtime.handle, ptr @block_has_used_kernel_address, ptr @block_has_used_kernel_address.runtime.handle, ptr @0, ptr @.runtime.handle, ptr @1, ptr @.runtime.handle.1, ptr @kernel_linkonce_odr_block, ptr @kernel_linkonce_odr_block.runtime.handle], section "llvm.metadata"


define amdgpu_kernel void @non_caller(ptr addrspace(1) %a, i8 %b, ptr addrspace(1) %c, i64 %d) {
  ret void
}

define amdgpu_kernel void @caller(ptr addrspace(1) %a, i8 %b, ptr addrspace(1) %c, i64 %d) {
entry:
  %block = alloca <{ i32, i32, ptr addrspace(1), i8 }>, align 8, addrspace(5)
  %inst = alloca %struct.ndrange_t, align 4, addrspace(5)
  %block2 = alloca <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }>, align 8, addrspace(5)
  %inst3 = alloca %struct.ndrange_t, align 4, addrspace(5)
  %block.size = getelementptr inbounds <{ i32, i32, ptr addrspace(1), i8 }>, ptr addrspace(5) %block, i32 0, i32 0
  store i32 25, ptr addrspace(5) %block.size, align 8
  %block.align = getelementptr inbounds <{ i32, i32, ptr addrspace(1), i8 }>, ptr addrspace(5) %block, i32 0, i32 1
  store i32 8, ptr addrspace(5) %block.align, align 4
  %block.captured = getelementptr inbounds <{ i32, i32, ptr addrspace(1), i8 }>, ptr addrspace(5) %block, i32 0, i32 2
  store ptr addrspace(1) %a, ptr addrspace(5) %block.captured, align 8
  %block.captured1 = getelementptr inbounds <{ i32, i32, ptr addrspace(1), i8 }>, ptr addrspace(5) %block, i32 0, i32 3
  store i8 %b, ptr addrspace(5) %block.captured1, align 8
  %inst4 = addrspacecast ptr addrspace(5) %block to ptr
  %inst5 = call i32 @__enqueue_kernel_basic(ptr addrspace(1) poison, i32 0, ptr addrspace(5) byval(%struct.ndrange_t) nonnull %inst,
  ptr @__test_block_invoke_kernel, ptr nonnull %inst4) #2
  %inst10 = call i32 @__enqueue_kernel_basic(ptr addrspace(1) poison, i32 0, ptr addrspace(5) byval(%struct.ndrange_t) nonnull %inst,
  ptr @__test_block_invoke_kernel, ptr nonnull %inst4) #2
  %inst11 = call i32 @__enqueue_kernel_basic(ptr addrspace(1) poison, i32 0, ptr addrspace(5) byval(%struct.ndrange_t) nonnull %inst,
  ptr @0, ptr nonnull %inst4) #2
  %inst12 = call i32 @__enqueue_kernel_basic(ptr addrspace(1) poison, i32 0, ptr addrspace(5) byval(%struct.ndrange_t) nonnull %inst,
  ptr @1, ptr nonnull %inst4) #2
  %block.size4 = getelementptr inbounds <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }>, ptr addrspace(5) %block2, i32 0, i32 0
  store i32 41, ptr addrspace(5) %block.size4, align 8
  %block.align5 = getelementptr inbounds <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }>, ptr addrspace(5) %block2, i32 0, i32 1
  store i32 8, ptr addrspace(5) %block.align5, align 4
  %block.captured7 = getelementptr inbounds <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }>, ptr addrspace(5) %block2, i32 0, i32 2
  store ptr addrspace(1) %a, ptr addrspace(5) %block.captured7, align 8
  %block.captured8 = getelementptr inbounds <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }>, ptr addrspace(5) %block2, i32 0, i32 5
  store i8 %b, ptr addrspace(5) %block.captured8, align 8
  %block.captured9 = getelementptr inbounds <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }>, ptr addrspace(5) %block2, i32 0, i32 3
  store ptr addrspace(1) %c, ptr addrspace(5) %block.captured9, align 8
  %block.captured10 = getelementptr inbounds <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }>, ptr addrspace(5) %block2, i32 0, i32 4
  store i64 %d, ptr addrspace(5) %block.captured10, align 8
  %inst8 = addrspacecast ptr addrspace(5) %block2 to ptr
  %inst9 = call i32 @__enqueue_kernel_basic(ptr addrspace(1) poison, i32 0, ptr addrspace(5) byval(%struct.ndrange_t) nonnull %inst3,
  ptr @__test_block_invoke_2_kernel, ptr nonnull %inst8) #2
  ret void
}

; __enqueue_kernel* functions may get inlined
define amdgpu_kernel void @inlined_caller(ptr addrspace(1) %a, i8 %b, ptr addrspace(1) %c, i64 %d) {
entry:
  %inst = load i64, ptr addrspace(1) addrspacecast (ptr @__test_block_invoke_kernel to ptr addrspace(1))
  store i64 %inst, ptr addrspace(1) %c
  ret void
}

; CHECK: define internal amdgpu_kernel void @__test_block_invoke_kernel(<{ i32, i32, ptr addrspace(1), i8 }> %arg) !associated !0 {
define internal amdgpu_kernel void @__test_block_invoke_kernel(<{ i32, i32, ptr addrspace(1), i8 }> %arg) #0 {
entry:
  %.fca.3.extract = extractvalue <{ i32, i32, ptr addrspace(1), i8 }> %arg, 2
  %.fca.4.extract = extractvalue <{ i32, i32, ptr addrspace(1), i8 }> %arg, 3
  store i8 %.fca.4.extract, ptr addrspace(1) %.fca.3.extract, align 1
  ret void
}

declare i32 @__enqueue_kernel_basic(ptr addrspace(1), i32, ptr addrspace(5), ptr, ptr) local_unnamed_addr

; CHECK: define internal amdgpu_kernel void @__test_block_invoke_2_kernel(<{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }> %arg) !associated !1 {
define internal amdgpu_kernel void @__test_block_invoke_2_kernel(<{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }> %arg) #0 {
entry:
  %.fca.3.extract = extractvalue <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }> %arg, 2
  %.fca.4.extract = extractvalue <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }> %arg, 3
  %.fca.5.extract = extractvalue <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }> %arg, 4
  %.fca.6.extract = extractvalue <{ i32, i32, ptr addrspace(1), ptr addrspace(1), i64, i8 }> %arg, 5
  store i8 %.fca.6.extract, ptr addrspace(1) %.fca.3.extract, align 1
  store i64 %.fca.5.extract, ptr addrspace(1) %.fca.4.extract, align 8
  ret void
}

@kernel_address_user = global [1 x ptr] [ ptr @block_has_used_kernel_address ]

; CHECK: define internal amdgpu_kernel void @block_has_used_kernel_address(<{ i32, i32, ptr addrspace(1), i8 }> %arg) !associated !2 {
define internal amdgpu_kernel void @block_has_used_kernel_address(<{ i32, i32, ptr addrspace(1), i8 }> %arg) #0 {
entry:
  %.fca.3.extract = extractvalue <{ i32, i32, ptr addrspace(1), i8 }> %arg, 2
  %.fca.4.extract = extractvalue <{ i32, i32, ptr addrspace(1), i8 }> %arg, 3
  store i8 %.fca.4.extract, ptr addrspace(1) %.fca.3.extract, align 1
  ret void
}

define amdgpu_kernel void @user_of_kernel_address(ptr addrspace(1) %arg) {
  store ptr @block_has_used_kernel_address, ptr addrspace(1) %arg
  ret void
}

; CHECK: define internal amdgpu_kernel void @0(<{ i32, i32, ptr addrspace(1), i8 }> %arg) !associated !3 {
define internal amdgpu_kernel void @0(<{ i32, i32, ptr addrspace(1), i8 }> %arg) #0 {
  ret void
}

; CHECK: define internal amdgpu_kernel void @1(<{ i32, i32, ptr addrspace(1), i8 }> %arg) !associated !4 {
define internal amdgpu_kernel void @1(<{ i32, i32, ptr addrspace(1), i8 }> %arg) #0 {
  ret void
}

; CHECK: define linkonce_odr amdgpu_kernel void @kernel_linkonce_odr_block() !associated !5 {
define linkonce_odr amdgpu_kernel void @kernel_linkonce_odr_block() #0 {
  ret void
}

attributes #0 = { "enqueued-block" }

; CHECK: !0 = !{ptr @__test_block_invoke_kernel.runtime.handle}
; CHECK: !1 = !{ptr @__test_block_invoke_2_kernel.runtime.handle}
; CHECK: !2 = !{ptr @block_has_used_kernel_address.runtime.handle}
; CHECK: !3 = !{ptr @.runtime.handle}
; CHECK: !4 = !{ptr @.runtime.handle.1}
; CHECK: !5 = !{ptr @kernel_linkonce_odr_block.runtime.handle}
