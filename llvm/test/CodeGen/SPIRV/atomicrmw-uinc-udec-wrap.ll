; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#PointerType:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#MemSem_SequentiallyConsistent:]] = OpConstant %[[#Int]] 16
; CHECK-DAG: %[[#Value:]] = OpConstant %[[#Int]] 42
; CHECK-DAG: %[[#One:]] = OpConstant %[[#Int]] 1
; CHECK-DAG: %[[#Scope_CrossDevice:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#Pointer:]] = OpVariable %[[#PointerType]] CrossWorkgroup

@ui = common dso_local addrspace(1) global i32 0, align 4

; CHECK:      %[[#Load:]] = OpLoad %[[#Int]] %[[#Pointer]] Aligned 4
; CHECK:      OpBranch %[[#Loop:]]
; CHECK:      %[[#Loop]] = OpLabel
; CHECK:      %[[#Phi:]] = OpPhi %[[#Int]] %[[#Load]] %[[#Entry:]] %[[#PhiNext:]] %[[#Loop]]
; CHECK:      %[[#Add:]] = OpIAdd %[[#Int]] %[[#Phi]] %[[#One]]
; CHECK:      %[[#GE:]] = OpUGreaterThanEqual %[[#Bool]] %[[#Phi]] %[[#Value]]
; CHECK:      %[[#Select:]] = OpSelect %[[#Int]] %[[#GE]] %[[#Scope_CrossDevice]] %[[#Add]]
; CHECK:      %[[#CmpXChg:]] = OpAtomicCompareExchange %[[#Int]] %[[#Ptr:]] %[[#Scope_CrossDevice]]
; CHECK-SAME: %[[#MemSem_SequentiallyConsistent]] %[[#MemSem_SequentiallyConsistent]] %[[#Select]] %[[#Phi]]
; CHECK:      %[[#Cond:]] = OpCompositeExtract %[[#Bool]] %[[#CmpXChgComposite:]] 1
; CHECK:      %[[#PhiNext]] = OpCompositeExtract %[[#Int]] %[[#CmpXChgComposite]] 0
; CHECK:      OpBranchConditional %[[#Cond]] %[[#Exit:]] %[[#Loop]]
; CHECK:      %[[#Exit]] = OpLabel

define dso_local spir_func void @atomicrmw_uinc_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}

; CHECK:      %[[#Load:]] = OpLoad %[[#Int]] %[[#Pointer]] Aligned 4
; CHECK:      OpBranch %[[#Loop:]]
; CHECK:      %[[#Loop]] = OpLabel
; CHECK:      %[[#Phi:]] = OpPhi %[[#Int]] %[[#Load]] %[[#Entry:]] %[[#PhiNext:]] %[[#Loop]]
; CHECK:      %[[#Sub:]] = OpISub %[[#Int]] %[[#Phi]] %[[#One]]
; CHECK:      %[[#Equal:]] = OpIEqual %[[#Bool]] %[[#Phi]] %[[#Scope_CrossDevice]]
; CHECK:      %[[#GT:]] = OpUGreaterThan %[[#Bool]] %[[#Phi]] %[[#Value]]
; CHECK:      %[[#Or:]] = OpLogicalOr %[[#Bool]] %[[#Equal]] %[[#GT]]
; CHECK:      %[[#Select:]] = OpSelect %[[#Int]] %[[#Or]] %[[#Value]] %[[#Sub]]
; CHECK:      %[[#CmpXChg:]] = OpAtomicCompareExchange %[[#Int]] %[[#Ptr:]] %[[#Scope_CrossDevice]] 
; CHECK-SAME: %[[#MemSem_SequentiallyConsistent]] %[[#MemSem_SequentiallyConsistent]] %[[#Select]] %[[#Phi]]
; CHECK:      %[[#Cond:]] = OpCompositeExtract %[[#Bool]] %[[#CmpXChgComposite:]] 1
; CHECK:      %[[#PhiNext]] = OpCompositeExtract %[[#Int]] %[[#CmpXChgComposite]] 0
; CHECK:      OpBranchConditional %[[#Cond]] %[[#Exit:]] %[[#Loop]]
; CHECK:      %[[#Exit]] = OpLabel

define dso_local spir_func void @atomicrmw_udec_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}
