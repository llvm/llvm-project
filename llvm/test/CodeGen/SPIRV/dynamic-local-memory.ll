; RUN: opt -S -mtriple=spirv64-intel -passes=spirv-lower-dynamic-local-memory %s | FileCheck %s --check-prefix=IR
; RUN: opt -S -mtriple=spirv64-amd-amdhsa -passes=spirv-lower-dynamic-local-memory %s | FileCheck %s --check-prefix=UNCHANGED
; RUN: opt -S -mtriple=spirv64-unknown-unknown -passes=spirv-lower-dynamic-local-memory %s | FileCheck %s --check-prefix=UNCHANGED
; RUN: llc -O0 -mtriple=spirv64-intel %s -o - | FileCheck %s --check-prefix=SPV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-intel %s -o - -filetype=obj | spirv-val %}

; Dynamic local memory, i.e. unsized external Workgroup arrays, is lowered to a
; trailing local pointer argument of every kernel. All declarations alias it.

@buf = external addrspace(3) global [0 x i32], align 4
@other = external addrspace(3) global [0 x i8], align 32
@unused = external addrspace(3) global [0 x i8]
@sized = internal addrspace(3) global [4 x i32] poison, align 4

; Only Intel GPUs take the memory as a kernel argument.
; UNCHANGED: @buf = external addrspace(3) global [0 x i32]
; UNCHANGED-NOT: @__spirv_dynamic_local_memory
; UNCHANGED: define spir_kernel void @kernel(ptr addrspace(1) noundef %out) addrspace(9)

; IR-NOT: @buf =
; IR-NOT: @other =
; IR: @unused = external addrspace(3) global [0 x i8]
; IR: @sized = internal addrspace(3) global [4 x i32] poison
; IR: @__spirv_dynamic_local_memory = internal unnamed_addr addrspace(3) global ptr addrspace(3) poison, align 8

; IR-LABEL: define spir_func i32 @load(
; IR: %[[PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR: getelementptr i32, ptr addrspace(3) %[[PTR]], i64 %i
define spir_func i32 @load(i64 %i) addrspace(9) {
  %p = getelementptr i32, ptr addrspace(3) @buf, i64 %i
  %v = load i32, ptr addrspace(3) %p
  ret i32 %v
}

; Uses in constant expressions become instructions.
; IR-LABEL: define spir_func ptr addrspace(4) @generic(
; IR: %[[PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR: %[[CAST:.*]] = addrspacecast ptr addrspace(3) %[[PTR]] to ptr addrspace(4)
; IR: ret ptr addrspace(4) %[[CAST]]
define spir_func ptr addrspace(4) @generic() addrspace(9) {
  ret ptr addrspace(4) addrspacecast (ptr addrspace(3) @other to ptr addrspace(4))
}

; Uses in PHIs load in the incoming block.
; IR-LABEL: define spir_func ptr addrspace(3) @phi(
; IR: then:
; IR-NEXT: %[[PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: br label %exit
; IR: phi ptr addrspace(3) [ %[[PTR]], %then ], [ @sized, %entry ]
define spir_func ptr addrspace(3) @phi(i1 %c) addrspace(9) {
entry:
  br i1 %c, label %then, label %exit
then:
  br label %exit
exit:
  %p = phi ptr addrspace(3) [ @buf, %then ], [ @sized, %entry ]
  ret ptr addrspace(3) %p
}

; Kernels get the argument, stored on entry after the allocas. Attributes and
; metadata are kept.
; IR-LABEL: define spir_kernel void @kernel(ptr addrspace(1) noundef %out, ptr addrspace(3) %dyn_local_mem) addrspace(9) #0 !kernel_arg_addr_space
; IR-NEXT: entry:
; IR-NEXT: %a = alloca i32
; IR-NEXT: store ptr addrspace(3) %dyn_local_mem, ptr addrspace(3) @__spirv_dynamic_local_memory
define spir_kernel void @kernel(ptr addrspace(1) noundef %out) addrspace(9) #0 !kernel_arg_addr_space !0 {
entry:
  %a = alloca i32
  %v = call spir_func addrspace(9) i32 @load(i64 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Kernels not accessing the memory get it too, so the runtime can treat all
; kernels of the module alike.
; IR-LABEL: define spir_kernel void @nouse(ptr addrspace(3) %dyn_local_mem)
; IR-NEXT: store ptr addrspace(3) %dyn_local_mem, ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: ret void
define spir_kernel void @nouse() addrspace(9) {
  ret void
}

; IR: attributes #0 = { "foo" }

; SPV-DAG: OpEntryPoint Kernel %[[#KERNEL:]] "kernel"
; SPV-DAG: OpEntryPoint Kernel %[[#NOUSE:]] "nouse"
; SPV-DAG: %[[#I8:]] = OpTypeInt 8 0
; SPV-DAG: %[[#LOCALPTR:]] = OpTypePointer Workgroup %[[#I8]]
; SPV-NOT: LinkageAttributes "buf"
; SPV-NOT: LinkageAttributes "other"
; SPV: %[[#KERNEL]] = OpFunction
; SPV-NEXT: OpFunctionParameter
; SPV-NEXT: OpFunctionParameter %[[#LOCALPTR]]
; SPV: %[[#NOUSE]] = OpFunction
; SPV-NEXT: OpFunctionParameter %[[#LOCALPTR]]

attributes #0 = { "foo" }

!0 = !{i32 1}
