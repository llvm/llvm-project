; Source
; typedef struct {int a;} ndrange_t;
;
; kernel void device_side_enqueue() {
;   ndrange_t ndrange;
;
;   get_kernel_work_group_size(^(){});
;   get_kernel_preferred_work_group_size_multiple(^(){});
;
; #pragma OPENCL EXTENSION cl_khr_subgroups : enable
;   get_kernel_max_sub_group_size_for_ndrange(ndrange, ^(){});
;   get_kernel_sub_group_count_for_ndrange(ndrange, ^(){});
; }
;
; Compilation command:
; clang -cc1 -triple spir-unknown-unknown -O0 -cl-std=CL2.0 -emit-llvm kernel_query.cl

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

source_filename = "kernel_query.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%struct.ndrange_t = type { i32 }

; CHECK-SPIRV: EntryPoint {{.*}} [[BlockKer1:[0-9]+]] "__device_side_enqueue_block_invoke_kernel"
; CHECK-SPIRV: EntryPoint {{.*}} [[BlockKer2:[0-9]+]] "__device_side_enqueue_block_invoke_2_kernel"
; CHECK-SPIRV: EntryPoint {{.*}} [[BlockKer3:[0-9]+]] "__device_side_enqueue_block_invoke_3_kernel"
; CHECK-SPIRV: EntryPoint {{.*}} [[BlockKer4:[0-9]+]] "__device_side_enqueue_block_invoke_4_kernel"
; CHECK-SPIRV: Name [[BlockGlb1:[0-9]+]] "__block_literal_global"
; CHECK-SPIRV: Name [[BlockGlb2:[0-9]+]] "__block_literal_global.1"
; CHECK-SPIRV: Name [[BlockGlb3:[0-9]+]] "__block_literal_global.2"
; CHECK-SPIRV: Name [[BlockGlb4:[0-9]+]] "__block_literal_global.3"

; CHECK-LLVM: [[BlockTy:%[0-9]+]] = type { i32, i32 }
%1 = type <{ i32, i32 }>

; CHECK-LLVM: @__block_literal_global = internal addrspace(1) constant [[BlockTy]] { i32 8, i32 4 }, align 4
; CHECK-LLVM: @__block_literal_global.1 = internal addrspace(1) constant [[BlockTy]] { i32 8, i32 4 }, align 4
; CHECK-LLVM: @__block_literal_global.2 = internal addrspace(1) constant [[BlockTy]] { i32 8, i32 4 }, align 4
; CHECK-LLVM: @__block_literal_global.3 = internal addrspace(1) constant [[BlockTy]] { i32 8, i32 4 }, align 4

@__block_literal_global = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4
@__block_literal_global.1 = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4
@__block_literal_global.2 = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4
@__block_literal_global.3 = internal addrspace(1) constant { i32, i32 } { i32 8, i32 4 }, align 4

; CHECK-SPIRV: TypeInt [[Int32Ty:[0-9]+]] 32
; CHECK-SPIRV: TypeInt [[Int8Ty:[0-9]+]] 8
; CHECK-SPIRV: Constant [[Int32Ty]] [[ConstInt8:[0-9]+]] 8
; CHECK-SPIRV: TypeVoid [[VoidTy:[0-9]+]]
; CHECK-SPIRV: TypeStruct [[NDRangeTy:[0-9]+]] [[Int32Ty]] {{$}}
; CHECK-SPIRV: TypePointer [[NDRangePtrTy:[0-9]+]] 7 [[NDRangeTy]]
; CHECK-SPIRV: TypePointer [[Int8PtrGenTy:[0-9]+]] 8 [[Int8Ty]]
; CHECK-SPIRV: TypeFunction [[BlockKerTy:[0-9]+]] [[VoidTy]] [[Int8PtrGenTy]]

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @device_side_enqueue() #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 {
entry:

; CHECK-SPIRV: Variable [[NDRangePtrTy]] [[NDRange:[0-9]+]]

  %ndrange = alloca %struct.ndrange_t, align 4

; CHECK-SPIRV: Bitcast {{[0-9]+}} [[BlockLit1Tmp:[0-9]+]] [[BlockGlb1]]
; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit1:[0-9]+]] [[BlockLit1Tmp]]
; CHECK-SPIRV: GetKernelWorkGroupSize [[Int32Ty]] {{[0-9]+}} [[BlockKer1]] [[BlockLit1]] [[ConstInt8]] [[ConstInt8]]

; CHECK-LLVM: call i32 @__get_kernel_work_group_size_impl(i8 addrspace(4)* {{.*}}, i8 addrspace(4)* {{.*}})

  %0 = call i32 @__get_kernel_work_group_size_impl(i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*))

; CHECK-SPIRV: Bitcast {{[0-9]+}} [[BlockLit2Tmp:[0-9]+]] [[BlockGlb2]]
; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit2:[0-9]+]] [[BlockLit2Tmp]]
; CHECK-SPIRV: GetKernelPreferredWorkGroupSizeMultiple [[Int32Ty]] {{[0-9]+}} [[BlockKer2]] [[BlockLit2]] [[ConstInt8]] [[ConstInt8]]

; CHECK-LLVM: call i32 @__get_kernel_preferred_work_group_size_multiple_impl(i8 addrspace(4)* {{.*}}, i8 addrspace(4)* {{.*}}) #1

  %1 = call i32 @__get_kernel_preferred_work_group_size_multiple_impl(i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_2_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global.1 to i8 addrspace(1)*) to i8 addrspace(4)*))

; CHECK-SPIRV: Bitcast {{[0-9]+}} [[BlockLit3Tmp:[0-9]+]] [[BlockGlb3]]
; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit3:[0-9]+]] [[BlockLit3Tmp]]
; CHECK-SPIRV: GetKernelNDrangeMaxSubGroupSize [[Int32Ty]] {{[0-9]+}} [[NDRange]] [[BlockKer3]] [[BlockLit3]] [[ConstInt8]] [[ConstInt8]]

; CHECK-LLVM: call i32 @__get_kernel_max_sub_group_size_for_ndrange_impl(%struct.ndrange_t* {{.*}}, i8 addrspace(4)* {{.*}}, i8 addrspace(4)* {{.*}})

  %2 = call i32 @__get_kernel_max_sub_group_size_for_ndrange_impl(%struct.ndrange_t* %ndrange, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_3_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global.2 to i8 addrspace(1)*) to i8 addrspace(4)*))

; CHECK-SPIRV: Bitcast {{[0-9]+}} [[BlockLit4Tmp:[0-9]+]] [[BlockGlb4]]
; CHECK-SPIRV: PtrCastToGeneric [[Int8PtrGenTy]] [[BlockLit4:[0-9]+]] [[BlockLit4Tmp]]
; CHECK-SPIRV: GetKernelNDrangeSubGroupCount [[Int32Ty]] {{[0-9]+}} [[NDRange]] [[BlockKer4]] [[BlockLit4]] [[ConstInt8]] [[ConstInt8]]

; CHECK-LLVM: call i32 @__get_kernel_sub_group_count_for_ndrange_impl(%struct.ndrange_t* {{.*}}, i8 addrspace(4)* {{.*}}, i8 addrspace(4)* {{.*}})

  %3 = call i32 @__get_kernel_sub_group_count_for_ndrange_impl(%struct.ndrange_t* %ndrange, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*)* @__device_side_enqueue_block_invoke_4_kernel to i8*) to i8 addrspace(4)*), i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32 } addrspace(1)* @__block_literal_global.3 to i8 addrspace(1)*) to i8 addrspace(4)*))
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke(i8 addrspace(4)* %.block_descriptor) #1 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_kernel(i8 addrspace(4)*) #2 {
entry:
  call void @__device_side_enqueue_block_invoke(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__get_kernel_work_group_size_impl(i8 addrspace(4)*, i8 addrspace(4)*)

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke_2(i8 addrspace(4)* %.block_descriptor) #1 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_2_kernel(i8 addrspace(4)*) #2 {
entry:
  call void @__device_side_enqueue_block_invoke_2(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__get_kernel_preferred_work_group_size_multiple_impl(i8 addrspace(4)*, i8 addrspace(4)*)

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke_3(i8 addrspace(4)* %.block_descriptor) #1 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_3_kernel(i8 addrspace(4)*) #2 {
entry:
  call void @__device_side_enqueue_block_invoke_3(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__get_kernel_max_sub_group_size_for_ndrange_impl(%struct.ndrange_t*, i8 addrspace(4)*, i8 addrspace(4)*)

; Function Attrs: convergent noinline nounwind optnone
define internal spir_func void @__device_side_enqueue_block_invoke_4(i8 addrspace(4)* %.block_descriptor) #1 {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  %block.addr = alloca <{ i32, i32 }> addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32 }> addrspace(4)*
  store <{ i32, i32 }> addrspace(4)* %block, <{ i32, i32 }> addrspace(4)** %block.addr, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_kernel void @__device_side_enqueue_block_invoke_4_kernel(i8 addrspace(4)*) #2 {
entry:
  call void @__device_side_enqueue_block_invoke_4(i8 addrspace(4)* %0)
  ret void
}

declare i32 @__get_kernel_sub_group_count_for_ndrange_impl(%struct.ndrange_t*, i8 addrspace(4)*, i8 addrspace(4)*)

; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer1]] 0 [[BlockKerTy]]
; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer2]] 0 [[BlockKerTy]]
; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer3]] 0 [[BlockKerTy]]
; CHECK-SPIRV-DAG: Function [[VoidTy]] [[BlockKer4]] 0 [[BlockKerTy]]

; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_kernel(i8 addrspace(4)*)
; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_2_kernel(i8 addrspace(4)*)
; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_3_kernel(i8 addrspace(4)*)
; CHECK-LLVM-DAG: define spir_kernel void @__device_side_enqueue_block_invoke_4_kernel(i8 addrspace(4)*)

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
