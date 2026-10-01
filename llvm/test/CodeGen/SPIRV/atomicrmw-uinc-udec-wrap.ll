; Verify that on AMD targets atomicrmw uinc_wrap/udec_wrap lower to
; OpFunctionCall to
; __translate_spirv_atomic_uinc_wrap_*/__translate_spirv_atomic_udec_wrap_* with
; Import linkage, rather than being expanded to a CmpXChg loop. The name carries
; a _p<addrspace>_i<width> suffix, because a module may need several mutually
; incompatible signatures while SPIR-V resolves an imported function by its
; linkage name alone.
;
; The helper is an AMD extension, so non-AMD targets keep the generic expansion
; instead; that is covered by atomicrmw-uinc-udec-wrap-non-amd.ll.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; AMD targets enable SPV_KHR_untyped_pointers by default, so pointers are untyped.
; CHECK-DAG: %[[#PointerType:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#MemSem_SequentiallyConsistent:]] = OpConstant %[[#Int]] 528
; CHECK-DAG: %[[#Value:]] = OpConstant %[[#Int]] 42
; CHECK-DAG: %[[#Scope_CrossDevice:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#Pointer:]] = OpUntypedVariableKHR %[[#PointerType]] CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#AllOnes:]] = OpConstant %[[#Int]] 4294967295

; CHECK-DAG: OpDecorate %[[#UIncWrapFn:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i32" Import
; CHECK-DAG: OpDecorate %[[#UDecWrapFn:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_i32" Import

@ui = common dso_local addrspace(1) global i32 0, align 4

; CHECK: OpFunctionCall %[[#Int]] %[[#UIncWrapFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_SequentiallyConsistent]] %[[#Value]]
define dso_local spir_func void @atomicrmw_uinc_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}

; CHECK: OpFunctionCall %[[#Int]] %[[#UDecWrapFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_SequentiallyConsistent]] %[[#Value]]
define dso_local spir_func void @atomicrmw_udec_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}

; CHECK:      %[[#Load:]] = OpLoad %[[#Int]] %[[#Pointer]] Aligned 4
; CHECK:      OpBranch %[[#Loop:]]
; CHECK:      %[[#Loop]] = OpLabel
; CHECK:      %[[#Phi:]] = OpPhi %[[#Int]] %[[#Load]] %[[#Entry:]] %[[#PhiNext:]] %[[#Loop]]
; CHECK:      %[[#And:]] = OpBitwiseAnd %[[#Int]] %[[#Phi]] %[[#Value]]
; CHECK:      %[[#Select:]] = OpBitwiseXor %[[#Int]] %[[#And]] %[[#AllOnes]]
; CHECK:      %[[#CmpXChg:]] = OpAtomicCompareExchange %[[#Int]] %[[#Ptr:]] %[[#Scope_CrossDevice]]
; CHECK-SAME: %[[#MemSem_SequentiallyConsistent]] %[[#MemSem_SequentiallyConsistent]] %[[#Select]] %[[#Phi]]
; CHECK:      %[[#Cond:]] = OpCompositeExtract %[[#Bool]] %[[#CmpXChgComposite:]] 1
; CHECK:      %[[#PhiNext]] = OpCompositeExtract %[[#Int]] %[[#CmpXChgComposite]] 0
; CHECK:      OpBranchConditional %[[#Cond]] %[[#Exit:]] %[[#Loop]]
; CHECK:      %[[#Exit]] = OpLabel

define dso_local spir_func void @atomicrmw_nand() local_unnamed_addr {
entry:
  %0 = atomicrmw nand ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}
