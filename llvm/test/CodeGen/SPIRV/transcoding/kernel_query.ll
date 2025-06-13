; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%struct.ndrange_t = type { i32 }
%1 = type <{ i32, i32 }>

@__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4
@__block_literal_global.1 = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4

; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#Int32Ty]] 4
; CHECK-DAG: %[[#C8:]] = OpConstant %[[#Int32Ty]] 8
; CHECK-DAG: %[[#NDRangeTy:]] = OpTypeStruct %[[#Int32Ty]]
; CHECK-DAG: %[[#NDRangePtrTy:]] = OpTypePointer Function %[[#NDRangeTy]]

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @device_side_enqueue() #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 {
entry:

; CHECK: %[[#NDRange:]] = OpVariable %[[#NDRangePtrTy]]

  %ndrange = alloca %struct.ndrange_t, align 4

; CHECK: %[[#BlockLit1:]] = OpPtrCastToGeneric %[[#]] %[[#]]
; CHECK: %[[#]] = OpGetKernelNDrangeMaxSubGroupSize %[[#Int32Ty]] %[[#NDRange]] %[[#]] %[[#BlockLit1]] %[[#C8]] %[[#C4]]

  %0 = call i32 @__get_kernel_max_sub_group_size_for_ndrange_impl(ptr %ndrange, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global to ptr addrspace(4)))

; CHECK: %[[#BlockLit2:]] = OpPtrCastToGeneric %[[#]] %[[#]]
; CHECK: %[[#]] = OpGetKernelNDrangeSubGroupCount %[[#Int32Ty]] %[[#NDRange]] %[[#]] %[[#BlockLit2]] %[[#C8]] %[[#C4]]

  %1 = call i32 @__get_kernel_sub_group_count_for_ndrange_impl(ptr %ndrange, ptr addrspace(4) addrspacecast (ptr @__device_side_enqueue_block_invoke_1_kernel to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global.1 to ptr addrspace(4)))
  ret void
}

declare i32 @__get_kernel_preferred_work_group_size_multiple_impl(ptr addrspace(4), ptr addrspace(4))

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke(ptr addrspace(4) %.block_descriptor) #1 {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  %block.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  store ptr addrspace(4) %.block_descriptor, ptr %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_kernel(ptr addrspace(4)) #2 {
entry:
  call void @__device_side_enqueue_block_invoke(ptr addrspace(4) %0)
  ret void
}

declare i32 @__get_kernel_max_sub_group_size_for_ndrange_impl(ptr, ptr addrspace(4), ptr addrspace(4))

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke_1(ptr addrspace(4) %.block_descriptor) #1 {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  %block.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  store ptr addrspace(4) %.block_descriptor, ptr %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_1_kernel(ptr addrspace(4)) #2 {
entry:
  call void @__device_side_enqueue_block_invoke_1(ptr addrspace(4) %0)
  ret void
}

declare i32 @__get_kernel_sub_group_count_for_ndrange_impl(ptr, ptr addrspace(4), ptr addrspace(4))

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { argmemonly nounwind }

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
