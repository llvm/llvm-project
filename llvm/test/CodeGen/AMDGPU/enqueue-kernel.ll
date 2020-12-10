; RUN: opt -data-layout=A5 -amdgpu-lower-enqueued-block -S < %s | FileCheck %s

; CHECK: @__test_block_invoke_kernel.runtime_handle = addrspace(1) global [2 x i64] zeroinitializer
; CHECK: @__test_block_invoke_2_kernel.runtime_handle = addrspace(1) global [2 x i64] zeroinitializer
; CHECK: @__amdgpu_enqueued_kernel.runtime_handle = addrspace(1) global [2 x i64] zeroinitializer
; CHECK: @__amdgpu_enqueued_kernel.1.runtime_handle = addrspace(1) global [2 x i64] zeroinitializer

%struct.ndrange_t = type { i32 }
%opencl.queue_t = type opaque

; CHECK-LABEL: define amdgpu_kernel void @non_caller
; CHECK-NOT: #{{[0-9]+}}
define amdgpu_kernel void @non_caller(i8 addrspace(1)* %a, i8 %b, i64 addrspace(1)* %c, i64 %d) local_unnamed_addr
  !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @caller_indirect
; CHECK-SAME: #[[AT_CALLER:[0-9]+]]
define amdgpu_kernel void @caller_indirect(i8 addrspace(1)* %a, i8 %b, i64 addrspace(1)* %c, i64 %d) local_unnamed_addr
  !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  call void @caller(i8 addrspace(1)* %a, i8 %b, i64 addrspace(1)* %c, i64 %d)
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @caller
; CHECK-SAME: #[[AT_CALLER]]
; CHECK-NOT: @__test_block_invoke_kernel
; CHECK-NOT: @__test_block_invoke_2_kernel
; CHECK-NOT: @__amdgpu_enqueued_kernel
; CHECK-NOT: @__amdgpu_enqueued_kernel.1
; CHECK-NOT: @0
; CHECK-NOT: @1
; CHECK: call i32 @__enqueue_kernel_basic({{.*}}@__test_block_invoke_kernel.runtime_handle
; CHECK: call i32 @__enqueue_kernel_basic({{.*}}@__test_block_invoke_kernel.runtime_handle
; CHECK: call i32 @__enqueue_kernel_basic({{.*}}@__amdgpu_enqueued_kernel.runtime_handle
; CHECK: call i32 @__enqueue_kernel_basic({{.*}}@__amdgpu_enqueued_kernel.1.runtime_handle
; CHECK: call i32 @__enqueue_kernel_basic({{.*}}@__test_block_invoke_2_kernel.runtime_handle
define amdgpu_kernel void @caller(i8 addrspace(1)* %a, i8 %b, i64 addrspace(1)* %c, i64 %d) local_unnamed_addr
  !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %block = alloca <{ i32, i32, i8 addrspace(1)*, i8 }>, align 8, addrspace(5)
  %tmp = alloca %struct.ndrange_t, align 4, addrspace(5)
  %block2 = alloca <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, align 8, addrspace(5)
  %tmp3 = alloca %struct.ndrange_t, align 4, addrspace(5)
  %block.size = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i8 }>, <{ i32, i32, i8 addrspace(1)*, i8 }> addrspace(5)* %block, i32 0, i32 0
  store i32 25, i32 addrspace(5)* %block.size, align 8
  %block.align = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i8 }>, <{ i32, i32, i8 addrspace(1)*, i8 }> addrspace(5)* %block, i32 0, i32 1
  store i32 8, i32 addrspace(5)* %block.align, align 4
  %block.captured = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i8 }>, <{ i32, i32, i8 addrspace(1)*, i8 }> addrspace(5)* %block, i32 0, i32 2
  store i8 addrspace(1)* %a, i8 addrspace(1)* addrspace(5)* %block.captured, align 8
  %block.captured1 = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i8 }>, <{ i32, i32, i8 addrspace(1)*, i8 }> addrspace(5)* %block, i32 0, i32 3
  store i8 %b, i8 addrspace(5)* %block.captured1, align 8
  %tmp1 = bitcast <{ i32, i32, i8 addrspace(1)*, i8 }> addrspace(5)* %block to void () addrspace(5)*
  %tmp4 = addrspacecast void () addrspace(5)* %tmp1 to i8*
  %tmp5 = call i32 @__enqueue_kernel_basic(%opencl.queue_t addrspace(1)* undef, i32 0, %struct.ndrange_t addrspace(5)* byval(%struct.ndrange_t) nonnull %tmp,
    i8* bitcast (void (<{ i32, i32, i8 addrspace(1)*, i8 }>)* @__test_block_invoke_kernel to i8*), i8* nonnull %tmp4) #2
  %tmp10 = call i32 @__enqueue_kernel_basic(%opencl.queue_t addrspace(1)* undef, i32 0, %struct.ndrange_t addrspace(5)* byval(%struct.ndrange_t) nonnull %tmp,
    i8* bitcast (void (<{ i32, i32, i8 addrspace(1)*, i8 }>)* @__test_block_invoke_kernel to i8*), i8* nonnull %tmp4) #2
  %tmp11 = call i32 @__enqueue_kernel_basic(%opencl.queue_t addrspace(1)* undef, i32 0, %struct.ndrange_t addrspace(5)* byval(%struct.ndrange_t) nonnull %tmp,
    i8* bitcast (void (<{ i32, i32, i8 addrspace(1)*, i8 }>)* @0 to i8*), i8* nonnull %tmp4) #2
  %tmp12 = call i32 @__enqueue_kernel_basic(%opencl.queue_t addrspace(1)* undef, i32 0, %struct.ndrange_t addrspace(5)* byval(%struct.ndrange_t) nonnull %tmp,
    i8* bitcast (void (<{ i32, i32, i8 addrspace(1)*, i8 }>)* @1 to i8*), i8* nonnull %tmp4) #2
  %block.size4 = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> addrspace(5)* %block2, i32 0, i32 0
  store i32 41, i32 addrspace(5)* %block.size4, align 8
  %block.align5 = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> addrspace(5)* %block2, i32 0, i32 1
  store i32 8, i32 addrspace(5)* %block.align5, align 4
  %block.captured7 = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> addrspace(5)* %block2, i32 0, i32 2
  store i8 addrspace(1)* %a, i8 addrspace(1)* addrspace(5)* %block.captured7, align 8
  %block.captured8 = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> addrspace(5)* %block2, i32 0, i32 5
  store i8 %b, i8 addrspace(5)* %block.captured8, align 8
  %block.captured9 = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> addrspace(5)* %block2, i32 0, i32 3
  store i64 addrspace(1)* %c, i64 addrspace(1)* addrspace(5)* %block.captured9, align 8
  %block.captured10 = getelementptr inbounds <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>, <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> addrspace(5)* %block2, i32 0, i32 4
  store i64 %d, i64 addrspace(5)* %block.captured10, align 8
  %tmp6 = bitcast <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> addrspace(5)* %block2 to void () addrspace(5)*
  %tmp8 = addrspacecast void () addrspace(5)* %tmp6 to i8*
  %tmp9 = call i32 @__enqueue_kernel_basic(%opencl.queue_t addrspace(1)* undef, i32 0, %struct.ndrange_t addrspace(5)* byval(%struct.ndrange_t) nonnull %tmp3,
    i8* bitcast (void (<{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }>)* @__test_block_invoke_2_kernel to i8*), i8* nonnull %tmp8) #2
  ret void
}

; __enqueue_kernel* functions may get inlined
; CHECK-LABEL: define amdgpu_kernel void @inlined_caller
; CHECK-SAME: #[[AT_CALLER]]
; CHECK-NOT: @__test_block_invoke_kernel
; CHECK: load i64, i64 addrspace(1)* getelementptr inbounds ([2 x i64], [2 x i64] addrspace(1)* @__test_block_invoke_kernel.runtime_handle, i32 0, i32 0)
define amdgpu_kernel void @inlined_caller(i8 addrspace(1)* %a, i8 %b, i64 addrspace(1)* %c, i64 %d) local_unnamed_addr
  !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %tmp = load i64, i64 addrspace(1)* addrspacecast (i64* bitcast (void (<{ i32, i32, i8 addrspace(1)*, i8 }>)* @__test_block_invoke_kernel to i64*) to i64 addrspace(1)*)
  store i64 %tmp, i64 addrspace(1)* %c
  ret void
}

; CHECK-LABEL: define dso_local amdgpu_kernel void @__test_block_invoke_kernel
; CHECK-SAME: #[[AT1:[0-9]+]]
define internal amdgpu_kernel void @__test_block_invoke_kernel(<{ i32, i32, i8 addrspace(1)*, i8 }> %arg) #0
  !kernel_arg_addr_space !14 !kernel_arg_access_qual !15 !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !17 {
entry:
  %.fca.3.extract = extractvalue <{ i32, i32, i8 addrspace(1)*, i8 }> %arg, 2
  %.fca.4.extract = extractvalue <{ i32, i32, i8 addrspace(1)*, i8 }> %arg, 3
  store i8 %.fca.4.extract, i8 addrspace(1)* %.fca.3.extract, align 1
  ret void
}

declare i32 @__enqueue_kernel_basic(%opencl.queue_t addrspace(1)*, i32, %struct.ndrange_t addrspace(5)*, i8*, i8*) local_unnamed_addr

; CHECK-LABEL: define dso_local amdgpu_kernel void @__test_block_invoke_2_kernel
; CHECK-SAME: #[[AT2:[0-9]+]]
define internal amdgpu_kernel void @__test_block_invoke_2_kernel(<{ i32, i32, i8 addrspace(1)*,
  i64 addrspace(1)*, i64, i8 }> %arg) #0 !kernel_arg_addr_space !14 !kernel_arg_access_qual !15
  !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !17 {
entry:
  %.fca.3.extract = extractvalue <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> %arg, 2
  %.fca.4.extract = extractvalue <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> %arg, 3
  %.fca.5.extract = extractvalue <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> %arg, 4
  %.fca.6.extract = extractvalue <{ i32, i32, i8 addrspace(1)*, i64 addrspace(1)*, i64, i8 }> %arg, 5
  store i8 %.fca.6.extract, i8 addrspace(1)* %.fca.3.extract, align 1
  store i64 %.fca.5.extract, i64 addrspace(1)* %.fca.4.extract, align 8
  ret void
}

; CHECK-LABEL: define dso_local amdgpu_kernel void @__amdgpu_enqueued_kernel
; CHECK-SAME: #[[AT3:[0-9]+]]
define internal amdgpu_kernel void @0(<{ i32, i32, i8 addrspace(1)*, i8 }> %arg) #0
  !kernel_arg_addr_space !14 !kernel_arg_access_qual !15 !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !17 {
  ret void
}

; CHECK-LABEL: define dso_local amdgpu_kernel void @__amdgpu_enqueued_kernel.1
; CHECK-SAME: #[[AT4:[0-9]+]]
define internal amdgpu_kernel void @1(<{ i32, i32, i8 addrspace(1)*, i8 }> %arg) #0
  !kernel_arg_addr_space !14 !kernel_arg_access_qual !15 !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !17 {
  ret void
}

; CHECK: attributes #[[AT_CALLER]] = { "calls-enqueue-kernel" }
; CHECK: attributes #[[AT1]] = {{.*}}"runtime-handle"="__test_block_invoke_kernel.runtime_handle"
; CHECK: attributes #[[AT2]] = {{.*}}"runtime-handle"="__test_block_invoke_2_kernel.runtime_handle"
; CHECK: attributes #[[AT3]] = {{.*}}"runtime-handle"="__amdgpu_enqueued_kernel.runtime_handle"
; CHECK: attributes #[[AT4]] = {{.*}}"runtime-handle"="__amdgpu_enqueued_kernel.1.runtime_handle"

attributes #0 = { "enqueued-block" }

!3 = !{i32 1, i32 0, i32 1, i32 0}
!4 = !{!"none", !"none", !"none", !"none"}
!5 = !{!"char*", !"char", !"long*", !"long"}
!6 = !{!"", !"", !"", !""}
!14 = !{i32 0}
!15 = !{!"none"}
!16 = !{!"__block_literal"}
!17 = !{!""}
