; RUN: opt -S -mtriple=spirv64-intel -passes=spirv-lower-dynamic-local-memory %s | FileCheck %s --check-prefix=IR
; RUN: opt -S -mtriple=spirv64-amd-amdhsa -passes=spirv-lower-dynamic-local-memory %s | FileCheck %s --check-prefix=UNCHANGED
; RUN: opt -S -mtriple=spirv64-unknown-unknown -passes=spirv-lower-dynamic-local-memory %s | FileCheck %s --check-prefix=IR
; RUN: sed '/^!llvm.module.flags/d' %s | opt -S -mtriple=spirv64-intel -passes=spirv-lower-dynamic-local-memory | FileCheck %s --check-prefix=UNCHANGED
; RUN: llc -O0 -mtriple=spirv64-intel %s -o - | FileCheck %s --check-prefix=SPV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-intel %s -o - -filetype=obj | spirv-val %}

; Dynamic local memory, i.e. unsized external Workgroup arrays, is lowered to a
; trailing local pointer argument of every kernel. All declarations alias it.

@buf = external addrspace(3) global [0 x i32], align 4
@other = external addrspace(3) global [0 x i8], align 32
@unused = external addrspace(3) global [0 x i8]
@sized = internal addrspace(3) global [4 x i32] poison, align 4

; AMDHSA and modules without the "spirv-dynamic-local-memory" flag should not
; be affected by the pass.
; UNCHANGED: @buf = external addrspace(3) global [0 x i32]
; UNCHANGED-NOT: @__spirv_dynamic_local_memory
; UNCHANGED: define spir_kernel void @kernel(ptr addrspace(1) noundef %out) addrspace(9)

; IR-NOT: @buf =
; IR-NOT: @other =
; IR: @unused = external addrspace(3) global [0 x i8]
; IR: @sized = internal addrspace(3) global [4 x i32] poison
; IR: @__spirv_dynamic_local_memory = internal unnamed_addr addrspace(3) global ptr addrspace(3) poison, align 8
; IR: @__spirv_dynamic_local_memory_present = addrspace(1) constant i8 1

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

; Uses in PHIs are also valid
; IR-LABEL: define spir_func ptr addrspace(3) @phi(
; IR: entry
; IR-NEXT: %[[PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR: then:
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

; IR-LABEL: define spir_func ptr addrspace(3) @phi_duplicate_predecessor(
; IR: entry:
; IR-NEXT: %[[PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR: exit:
; IR-NEXT: %p = phi ptr addrspace(3) [ %[[PTR]], %entry ], [ %[[PTR]], %entry ], [ @sized, %other ]
define spir_func ptr addrspace(3) @phi_duplicate_predecessor(i32 %selector) addrspace(9) {
entry:
  switch i32 %selector, label %other [
    i32 0, label %exit
    i32 1, label %exit
  ]

other:
  br label %exit

exit:
  %p = phi ptr addrspace(3) [ @buf, %entry ], [ @buf, %entry ], [ @sized, %other ]
  ret ptr addrspace(3) %p
}

; Kernels get the argument, stored on entry after the allocas. Attributes and
; metadata are kept.
; IR-LABEL: define spir_kernel void @kernel(
; IR-SAME: ptr addrspace(1) noundef %out, ptr addrspace(3) %dyn_local_mem) addrspace(9) #[[KERNEL_ATTRS:[0-9]+]] !kernel_arg_addr_space
; IR-NEXT: entry:
; IR-NEXT: %a = alloca i32
; IR-NEXT: %[[X:.*]] = call{{( addrspace\(9\))?}} i64 @llvm.spv.thread.id.in.group.i64(i32 0)
; IR-NEXT: %[[XZERO:.*]] = icmp eq i64 %[[X]], 0
; IR-NEXT: %[[FIRSTX:.*]] = and i1 true, %[[XZERO]]
; IR-NEXT: %[[Y:.*]] = call{{( addrspace\(9\))?}} i64 @llvm.spv.thread.id.in.group.i64(i32 1)
; IR-NEXT: %[[YZERO:.*]] = icmp eq i64 %[[Y]], 0
; IR-NEXT: %[[FIRSTXY:.*]] = and i1 %[[FIRSTX]], %[[YZERO]]
; IR-NEXT: %[[Z:.*]] = call{{( addrspace\(9\))?}} i64 @llvm.spv.thread.id.in.group.i64(i32 2)
; IR-NEXT: %[[ZZERO:.*]] = icmp eq i64 %[[Z]], 0
; IR-NEXT: %[[FIRST:.*]] = and i1 %[[FIRSTXY]], %[[ZZERO]]
; IR-NEXT: br i1 %[[FIRST]], label %[[INIT:.*]], label %[[READY:.*]]
; IR: [[INIT]]:
; IR-NEXT: store ptr addrspace(3) %dyn_local_mem, ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: br label %[[READY]]
; IR: [[READY]]:
; IR-NEXT: call{{( addrspace\(9\))?}} void @llvm.spv.group.memory.barrier.with.group.sync()
; IR-NEXT: %v = call spir_func addrspace(9) i32 @load(i64 0)
define spir_kernel void @kernel(ptr addrspace(1) noundef %out) addrspace(9) #0 !kernel_arg_addr_space !0 {
entry:
  %a = alloca i32
  %v = call spir_func addrspace(9) i32 @load(i64 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Test ordering of loads relative to alloca in kernels using SLM
; IR-LABEL: define spir_kernel void @use_kernel_alloca(
; IR-NEXT: entry:
; IR-NEXT: %a = alloca i32
; IR-NEXT: %{{.*}} = call{{( addrspace\(9\))?}} i64 @llvm.spv.thread.id.in.group.i64(i32 0)
; IR: call{{( addrspace\(9\))?}} void @llvm.spv.group.memory.barrier.with.group.sync()
; IR-NEXT: %[[ALLOCA_LOCAL_PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: %p0 = getelementptr i32, ptr addrspace(3) %[[ALLOCA_LOCAL_PTR]], i64 0
define spir_kernel void @use_kernel_alloca(ptr addrspace(1) noundef %out) addrspace(9) #0 !kernel_arg_addr_space !0 {
entry:
  %a = alloca i32
  %p0 = getelementptr i32, ptr addrspace(3) @buf, i64 0
  %v0 = load i32, ptr addrspace(3) %p0
  store i32 %v0, ptr addrspace(1) %out
  ret void
}

; Kernels that use the memory load the slot once, after storing the argument,
; and reuse the loaded pointer for every access.
; IR-LABEL: define spir_kernel void @use_kernel(
; IR-SAME: ptr addrspace(1) noundef %out, ptr addrspace(3) %[[LOCAL_ARG:[a-zA-Z0-9_]+]]) addrspace(9)
; IR-NEXT: entry:
; IR: br i1 %{{.*}}, label %[[INIT:.*]], label %[[READY:.*]]
; IR: [[INIT]]:
; IR-NEXT: store ptr addrspace(3) %[[LOCAL_ARG]], ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: br label %[[READY]]
; IR: [[READY]]:
; IR-NEXT: call{{( addrspace\(9\))?}} void @llvm.spv.group.memory.barrier.with.group.sync()
; IR-NEXT: %[[LOCAL_PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: %p0 = getelementptr i32, ptr addrspace(3) %[[LOCAL_PTR]], i64 0
; IR-NEXT: %v0 = load i32, ptr addrspace(3) %p0
; IR-NEXT: store i32 %v0, ptr addrspace(1) %out
; IR-NEXT: %p1 = getelementptr i32, ptr addrspace(3) %[[LOCAL_PTR]], i64 1
; IR-NEXT: %v1 = load i32, ptr addrspace(3) %p1
; IR-NEXT: store i32 %v1, ptr addrspace(1) %out
; IR-NEXT: ret void
; IR-NEXT: }
define spir_kernel void @use_kernel(ptr addrspace(1) noundef %out) addrspace(9) #0 !kernel_arg_addr_space !0 {
entry:
  %p0 = getelementptr i32, ptr addrspace(3) @buf, i64 0
  %v0 = load i32, ptr addrspace(3) %p0
  store i32 %v0, ptr addrspace(1) %out
  %p1 = getelementptr i32, ptr addrspace(3) @buf, i64 1
  %v1 = load i32, ptr addrspace(3) %p1
  store i32 %v1, ptr addrspace(1) %out
  ret void
}

; Kernels not accessing the memory get it too, so the runtime can treat all
; kernels of the module alike.
; IR-LABEL: define spir_kernel void @nouse(ptr addrspace(3) %dyn_local_mem)
; IR: br i1 %{{.*}}, label %[[INIT:.*]], label %[[READY:.*]]
; IR: [[INIT]]:
; IR-NEXT: store ptr addrspace(3) %dyn_local_mem, ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: br label %[[READY]]
; IR: [[READY]]:
; IR-NEXT: call{{( addrspace\(9\))?}} void @llvm.spv.group.memory.barrier.with.group.sync()
; IR-NEXT: ret void
define spir_kernel void @nouse() addrspace(9) {
  ret void
}

; Functions should receive modified memory attributes
; IR-LABEL: define spir_func ptr addrspace(3) @nouse_function()
; IR-SAME: addrspace(9) #[[READ_ATTRS:[0-9]+]] {
; IR-NEXT: %[[PTR:.*]] = load ptr addrspace(3), ptr addrspace(3) @__spirv_dynamic_local_memory
; IR-NEXT: ret ptr addrspace(3) %[[PTR]]
define spir_func ptr addrspace(3) @nouse_function() addrspace(9) memory(none) {
  ret ptr addrspace(3) @buf
}

; Memory attributes of a caller must be updated
; IR-LABEL: define spir_func ptr addrspace(3) @caller()
; IR-SAME: addrspace(9) #[[READ_ATTRS]] {
; IR-NEXT: %[[PTR:.*]] = call spir_func addrspace(9) ptr addrspace(3) @nouse_function() #[[READ_ATTRS]]
; IR-NEXT: ret ptr addrspace(3) %[[PTR]]
define spir_func ptr addrspace(3) @caller() addrspace(9) memory(none) {
  %ptr = call spir_func addrspace(9) ptr addrspace(3) @nouse_function()
  ret ptr addrspace(3) %ptr
}

; Memory attributes of a whole call chain must be updated
; IR-LABEL: define spir_func ptr addrspace(3) @caller_chain()
; IR-SAME: addrspace(9) #[[READ_ATTRS]] {
; IR-NEXT: %[[PTR:.*]] = call spir_func addrspace(9) ptr addrspace(3) @caller() #[[READ_ATTRS]]
; IR-NEXT: ret ptr addrspace(3) %[[PTR]]
define spir_func ptr addrspace(3) @caller_chain() addrspace(9) memory(none) {
  %ptr = call spir_func addrspace(9) ptr addrspace(3) @caller()
  ret ptr addrspace(3) %ptr
}

; Memory attributes of a call instruction must be updated
; IR-LABEL: define spir_func ptr addrspace(3) @call_instruction()
; IR-SAME: addrspace(9) #[[READ_ATTRS]] {
; IR-NEXT: %[[PTR:.*]] = call spir_func addrspace(9) ptr addrspace(3) @caller() #[[READ_ATTRS]]
; IR-NEXT: ret ptr addrspace(3) %[[PTR]]
define spir_func ptr addrspace(3) @call_instruction() addrspace(9) memory(none) {
  %ptr = call spir_func addrspace(9) ptr addrspace(3) @caller() memory(none)
  ret ptr addrspace(3) %ptr
}

; IR: attributes #[[KERNEL_ATTRS]] = { memory(readwrite) "foo" }
; IR: attributes #[[READ_ATTRS]] = { memory(read, argmem: none, inaccessiblemem: none, errnomem: none, target_mem: none) }

; SPV-DAG: OpEntryPoint Kernel %[[#KERNEL:]] "kernel"
; SPV-DAG: OpEntryPoint Kernel %[[#NOUSE:]] "nouse"
; SPV-NOT: LinkageAttributes "buf"
; SPV-NOT: LinkageAttributes "other"
; SPV-DAG: OpDecorate %[[#MARKER:]] LinkageAttributes "__spirv_dynamic_local_memory_present" Export
; SPV-DAG: %[[#I8:]] = OpTypeInt 8 0
; SPV-DAG: %[[#LOCALPTR:]] = OpTypePointer Workgroup %[[#I8]]
; SPV-DAG: %[[#GLOBALPTR:]] = OpTypePointer CrossWorkgroup %[[#I8]]
; SPV-DAG: %[[#ONE:]] = OpConstant %[[#I8]] 1{{$}}
; SPV-DAG: %[[#MARKER]] = OpVariable %[[#GLOBALPTR]] CrossWorkgroup %[[#ONE]]
; SPV-DAG: %[[#I32:]] = OpTypeInt 32 0
; SPV-DAG: %[[#WORKGROUP:]] = OpConstant %[[#I32]] 2{{$}}
; SPV-DAG: %[[#ACQREL_WORKGROUP:]] = OpConstant %[[#I32]] 264{{$}}
; SPV: %[[#KERNEL]] = OpFunction
; SPV-NEXT: OpFunctionParameter
; SPV-NEXT: OpFunctionParameter %[[#LOCALPTR]]
; SPV: OpBranchConditional %{{[0-9]+}} %[[#INIT:]] %[[#READY:]]
; SPV: %[[#INIT]] = OpLabel
; SPV-NEXT: OpStore
; SPV-NEXT: OpBranch %[[#READY]]
; SPV: %[[#READY]] = OpLabel
; SPV-NEXT: OpControlBarrier %[[#WORKGROUP]] %[[#WORKGROUP]] %[[#ACQREL_WORKGROUP]]
; SPV: %[[#NOUSE]] = OpFunction
; SPV-NEXT: OpFunctionParameter %[[#LOCALPTR]]
; SPV: OpControlBarrier %[[#WORKGROUP]] %[[#WORKGROUP]] %[[#ACQREL_WORKGROUP]]

attributes #0 = { "foo" }

!0 = !{i32 1}

!llvm.module.flags = !{!1}
!1 = !{i32 7, !"spirv-dynamic-local-memory", i32 1}
