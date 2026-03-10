; RUN: llc -O0 -mtriple=spirv64-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-- %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-- %s -o - -filetype=obj | spirv-val %}

; Check that 'load atomic' LLVM IR instructions are lowered correctly to
; OpAtomicLoad with the right Scope and Memory Semantics operands.
;
; unordered and monotonic are currently mapped to Memory Semantics `None (Relaxed)` 0x0

; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#Int32]]
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#Int32]] 1{{$}}
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#Int32]] 2{{$}}
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Int32]] 3{{$}}
; CHECK-DAG: %[[#Const4:]] = OpConstant %[[#Int32]] 4{{$}}
; CHECK-DAG: %[[#Const16:]] = OpConstant %[[#Int32]] 16{{$}}

define i32 @load_i32_unordered(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const0]] %[[#Const0]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr unordered, align 4
  ret i32 %val
}

define i32 @load_i32_monotonic(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const0]] %[[#Const0]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr monotonic, align 4
  ret i32 %val
}

define i32 @load_i32_acquire(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const0]] %[[#Const2]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr acquire, align 4
  ret i32 %val
}

define i32 @load_i32_seq_cst(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const0]] %[[#Const16]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr seq_cst, align 4
  ret i32 %val
}

; -- test with different syncscopes 

define i32 @load_i32_acquire_singlethread(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const4]] %[[#Const2]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("singlethread") acquire, align 4
  ret i32 %val
}

define i32 @load_i32_acquire_subgroup(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const3]] %[[#Const2]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("subgroup") acquire, align 4
  ret i32 %val
}

define i32 @load_i32_acquire_workgroup(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const2]] %[[#Const2]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("workgroup") acquire, align 4
  ret i32 %val
}

define i32 @load_i32_acquire_device(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const1]] %[[#Const2]]
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("device") acquire, align 4
  ret i32 %val
}

; -- test with a different scalar type

define float @load_float_acquire(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#load:]] = OpAtomicLoad %[[#Int32]] %[[#ptr]] %[[#Const0]] %[[#Const2]]
; CHECK:       %[[#val:]] = OpBitcast %[[#Float]] %[[#load]]
; CHECK:       OpReturnValue %[[#val]]
  %val = load atomic float, ptr addrspace(1) %ptr acquire, align 8
  ret float %val
}
