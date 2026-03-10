; RUN: llc -O0 -mtriple=spirv64-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-- %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-- %s -o - -filetype=obj | spirv-val %}

; Check that 'store atomic' LLVM IR instructions are lowered correctly to
; OpAtomicStore with the right Scope and Memory Semantics operands.
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

define void @store_i32_unordered(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const0]] %[[#Const0]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr unordered, align 4
  ret void
}

define void @store_i32_monotonic(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const0]] %[[#Const0]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr monotonic, align 4
  ret void
}

define void @store_i32_release(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const0]] %[[#Const4]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr release, align 4
  ret void
}

define void @store_i32_seq_cst(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const0]] %[[#Const16]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr seq_cst, align 4
  ret void
}

; -- test with different syncscopes 

define void @store_i32_release_singlethread(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const4]] %[[#Const4]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("singlethread") release, align 4
  ret void
}

define void @store_i32_release_subgroup(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const3]] %[[#Const4]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("subgroup") release, align 4
  ret void
}

define void @store_i32_release_workgroup(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const2]] %[[#Const4]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("workgroup") release, align 4
  ret void
}

define void @store_i32_release_device(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const1]] %[[#Const4]] %[[#val]]
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("device") release, align 4
  ret void
}

; -- test with a different scalar type

define void @store_float_release(ptr addrspace(1) %ptr, float %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Float]]
; CHECK:       %[[#cast:]] = OpBitcast %[[#Int32]] %[[#val]]
; CHECK:       OpAtomicStore %[[#ptr]] %[[#Const0]] %[[#Const4]] %[[#cast]]
; CHECK:       OpReturn
  store atomic float %val, ptr addrspace(1) %ptr release, align 8
  ret void
}
