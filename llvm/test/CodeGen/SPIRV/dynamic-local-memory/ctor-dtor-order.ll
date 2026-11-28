; RUN: llc -O0 -mtriple=spirv64-intel --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-intel --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; Ctor/dtor lowering must create the init/fini kernels before dynamic local
; memory lowering adds a trailing local pointer argument to every kernel,
; including those that do not access dynamic local memory.

; CHECK-DAG: OpEntryPoint Kernel %[[#INIT:]] "spirv$device$init"
; CHECK-DAG: OpEntryPoint Kernel %[[#FINI:]] "spirv$device$fini"
; CHECK-DAG: OpName %[[#SLOT:]] "__spirv_dynamic_local_memory"
; CHECK-DAG: OpName %[[#CTOR:]] "ctor"
; CHECK-DAG: OpDecorate %[[#MARKER:]] LinkageAttributes "__spirv_dynamic_local_memory_present" Export
; CHECK-DAG: OpDecorate %[[#MARKER]] Constant
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#LOCALPTR:]] = OpTypePointer Workgroup %[[#I8]]
; CHECK-DAG: %[[#GLOBALPTR:]] = OpTypePointer CrossWorkgroup %[[#I8]]
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I8]] 1{{$}}
; CHECK-DAG: %[[#MARKER]] = OpVariable %[[#GLOBALPTR]] CrossWorkgroup %[[#ONE]]
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I32PTR:]] = OpTypePointer Workgroup %[[#I32]]
; CHECK-DAG: %[[#VALUE:]] = OpConstant %[[#I32]] 42{{$}}
; CHECK-DAG: %[[#WORKGROUP:]] = OpConstant %[[#I32]] 2{{$}}
; CHECK-DAG: %[[#ACQREL_WORKGROUP:]] = OpConstant %[[#I32]] 264{{$}}
; CHECK: %[[#CTOR]] = OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#LOCAL:]] = OpLoad %[[#LOCALPTR]] %[[#SLOT]] Aligned 8
; CHECK-NEXT: %[[#PTR:]] = OpBitcast %[[#I32PTR]] %[[#LOCAL]]
; CHECK-NEXT: OpStore %[[#PTR]] %[[#VALUE]] Aligned 4
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#INIT]] = OpFunction
; CHECK-NEXT: %[[#INIT_ARG:]] = OpFunctionParameter %[[#LOCALPTR]]
; CHECK-NEXT: OpLabel
; CHECK: OpBranchConditional %{{[0-9]+}} %[[#INIT_STORE:]] %[[#INIT_READY:]]
; CHECK-NEXT: %[[#INIT_STORE]] = OpLabel
; CHECK-NEXT: OpStore %[[#SLOT]] %[[#INIT_ARG]] Aligned 8
; CHECK-NEXT: OpBranch %[[#INIT_READY]]
; CHECK-NEXT: %[[#INIT_READY]] = OpLabel
; CHECK-NEXT: OpControlBarrier %[[#WORKGROUP]] %[[#WORKGROUP]] %[[#ACQREL_WORKGROUP]]
; CHECK: OpFunctionPointerCallINTEL
; CHECK: OpFunctionEnd
; CHECK: %[[#FINI]] = OpFunction
; CHECK-NEXT: %[[#FINI_ARG:]] = OpFunctionParameter %[[#LOCALPTR]]
; CHECK-NEXT: OpLabel
; CHECK: OpBranchConditional %{{[0-9]+}} %[[#FINI_STORE:]] %[[#FINI_READY:]]
; CHECK-NEXT: %[[#FINI_STORE]] = OpLabel
; CHECK-NEXT: OpStore %[[#SLOT]] %[[#FINI_ARG]] Aligned 8
; CHECK-NEXT: OpBranch %[[#FINI_READY]]
; CHECK-NEXT: %[[#FINI_READY]] = OpLabel
; CHECK-NEXT: OpControlBarrier %[[#WORKGROUP]] %[[#WORKGROUP]] %[[#ACQREL_WORKGROUP]]
; CHECK: OpFunctionPointerCallINTEL
; CHECK: OpFunctionEnd

@buf = external addrspace(3) global [0 x i32], align 4

@llvm.global_ctors = appending global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 65535, ptr addrspace(9) @ctor, ptr addrspace(9) null }
]
@llvm.global_dtors = appending global [1 x { i32, ptr addrspace(9), ptr addrspace(9) }] [
  { i32, ptr addrspace(9), ptr addrspace(9) } { i32 65535, ptr addrspace(9) @dtor, ptr addrspace(9) null }
]

define spir_kernel void @kernel(ptr addrspace(1) %out) addrspace(9) {
  %value = load i32, ptr addrspace(3) @buf
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; Constructor function accessing dynamic local memory should use slot
; initialized by init kernel.
define void @ctor() addrspace(9) {
  store i32 42, ptr addrspace(3) @buf, align 4
  ret void
}

define void @dtor() addrspace(9) {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 7, !"openmp-device", i32 50}
!1 = !{i32 7, !"spirv-dynamic-local-memory", i32 1}
