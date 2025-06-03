; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_arbitrary_precision_integers -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_INTEL_arbitrary_precision_integers -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT

; TODO: This test currently fails with LLVM_ENABLE_EXPENSIVE_CHECKS enabled
; XFAIL: expensive_checks

; CHECK-DAG: OpName %[[#Struct:]] "struct"
; CHECK-DAG: OpName %[[#Arg:]] "arg"
; CHECK-DAG: OpName %[[#QArg:]] "qarg"
; CHECK-DAG: OpName %[[#R:]] "r"
; CHECK-DAG: OpName %[[#Q:]] "q"
; CHECK-DAG: OpName %[[#Tr:]] "tr"
; CHECK-DAG: OpName %[[#Tq:]] "tq"
; CHECK-DAG: %[[#Struct]] = OpTypeStruct %[[#]] %[[#]] %[[#]]
; CHECK-DAG: %[[#PtrStruct:]] = OpTypePointer CrossWorkgroup %[[#Struct]]
; CHECK-EXT-DAG: %[[#Int40:]] = OpTypeInt 40 0
; CHECK-EXT-DAG: %[[#Int50:]] = OpTypeInt 50 0
; CHECK-NOEXT-DAG: %[[#Int40:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrInt40:]] = OpTypePointer CrossWorkgroup %[[#Int40]]

; CHECK: OpFunction

; CHECK-EXT: %[[#Tr]] = OpUConvert %[[#Int40]] %[[#R]]
; CHECK-EXT: %[[#Store:]] = OpInBoundsPtrAccessChain %[[#PtrStruct]] %[[#Arg]] %[[#]]
; CHECK-EXT: %[[#StoreAsInt40:]] = OpBitcast %[[#PtrInt40]] %[[#Store]]
; CHECK-EXT: OpStore %[[#StoreAsInt40]] %[[#Tr]]

; CHECK-NOEXT: %[[#Store:]] = OpInBoundsPtrAccessChain %[[#PtrStruct]] %[[#Arg]] %[[#]]
; CHECK-NOEXT: %[[#StoreAsInt40:]] = OpBitcast %[[#PtrInt40]] %[[#Store]]
; CHECK-NOEXT: OpStore %[[#StoreAsInt40]] %[[#R]]

; CHECK: OpFunction

; CHECK-EXT: %[[#Tq]] = OpUConvert %[[#Int40]] %[[#Q]]
; CHECK-EXT: OpStore %[[#QArg]] %[[#Tq]]

; CHECK-NOEXT: OpStore %[[#QArg]] %[[#Q]]

%struct = type <{ i32, i8, [3 x i8] }>

define spir_kernel void @foo(ptr addrspace(1) %arg, i64 %r) {
  %tr = trunc i64 %r to i40
  %addr = getelementptr inbounds %struct, ptr addrspace(1) %arg, i64 0
  store i40 %tr, ptr addrspace(1) %addr
  ret void
}

define spir_kernel void @bar(ptr addrspace(1) %qarg, i50 %q) {
  %tq = trunc i50 %q to i40
  store i40 %tq, ptr addrspace(1) %qarg
  ret void
}
