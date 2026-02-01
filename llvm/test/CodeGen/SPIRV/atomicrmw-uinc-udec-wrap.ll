; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Int]] %[[#Bool]]
; CHECK-DAG: %[[#StructPtr:]] = OpTypePointer CrossWorkgroup %[[#Struct]]
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#UndefStruct:]] = OpUndef %[[#Struct]]
; CHECK-DAG: %[[#MemSem:]] = OpConstant %[[#Int]] 16
; CHECK-DAG: %[[#Value:]] = OpConstant %[[#Int]] 42
; CHECK-DAG: %[[#One:]] = OpConstant %[[#Int]] 1
; CHECK-DAG: %[[#Scope:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#Pointer:]] = OpVariable %[[#IntPtr]] CrossWorkgroup

@ui = common dso_local addrspace(1) global i32 0, align 4

define dso_local spir_func void @atomicrmw_uinc_udec_wrap() local_unnamed_addr {
entry:
; CHECK: %[[#Entry:]] = OpLabel
; CHECK: %[[#Cast1:]] = OpBitcast %[[#StructPtr]] %[[#Pointer]]
; CHECK: %[[#Cast2:]] = OpBitcast %[[#StructPtr]] %[[#Pointer]]

  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst
; CHECK: %[[#LoadUi1:]] = OpLoad %[[#Int]] %[[#Pointer]] Aligned 4
; CHECK: OpBranch %[[#Loop1:]]
; CHECK: %[[#Loop1]] = OpLabel
; CHECK: %[[#Phi1:]] = OpPhi %[[#Int]] %[[#LoadUi1]] %[[#Entry]] %[[#Phi1Next:]] %[[#Loop1]]
; CHECK: %[[#Add:]] = OpIAdd %[[#Int]] %[[#Phi1]] %[[#One]]
; CHECK: %[[#GE:]] = OpUGreaterThanEqual %[[#Bool]] %[[#Phi1]] %[[#Value]]
; CHECK: %[[#Select1:]] = OpSelect %[[#Int]] %[[#GE]] %[[#Scope]] %[[#Add]]
; CHECK: %[[#Bitcast1:]] = OpBitcast %[[#IntPtr]] %[[#Cast1]]
; CHECK: %[[#CmpXChg1:]] = OpAtomicCompareExchange %[[#Int]] %[[#Bitcast1]] %[[#Scope]] %[[#MemSem]] %[[#MemSem]] %[[#Select1]] %[[#Phi1]]
; CHECK: %[[#Eq1:]] = OpIEqual %[[#Bool]] %[[#CmpXChg1]] %[[#Phi1]]
; CHECK: %[[#Insert1:]] = OpCompositeInsert %[[#Struct]] %[[#CmpXChg1]] %[[#UndefStruct]] 0
; CHECK: %[[#Insert2:]] = OpCompositeInsert %[[#Struct]] %[[#Eq1]] %[[#Insert1]] 1
; CHECK: %[[#Cond1:]] = OpCompositeExtract %[[#Bool]] %[[#Insert2]] 1
; CHECK: %[[#Phi1Next]] = OpCompositeExtract %[[#Int]] %[[#Insert2]] 0
; CHECK: OpBranchConditional %[[#Cond1]] %[[#Exit1:]] %[[#Loop1]]
; CHECK: %[[#Exit1]] = OpLabel

  %1 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
; CHECK: %[[#LoadUi2:]] = OpLoad %[[#Int]] %[[#Pointer]] Aligned 4
; CHECK: OpBranch %[[#Loop2:]]
; CHECK: %[[#Loop2]] = OpLabel
; CHECK: %[[#Phi2:]] = OpPhi %[[#Int]] %[[#LoadUi2]] %[[#Exit1]] %[[#Phi2Next:]] %[[#Loop2]]
; CHECK: %[[#Sub:]] = OpISub %[[#Int]] %[[#Phi2]] %[[#One]]
; CHECK: %[[#Eq2:]] = OpIEqual %[[#Bool]] %[[#Phi2]] %[[#Scope]]
; CHECK: %[[#GT:]] = OpUGreaterThan %[[#Bool]] %[[#Phi2]] %[[#Value]]
; CHECK: %[[#Or:]] = OpLogicalOr %[[#Bool]] %[[#Eq2]] %[[#GT]]
; CHECK: %[[#Select2:]] = OpSelect %[[#Int]] %[[#Or]] %[[#Value]] %[[#Sub]]
; CHECK: %[[#Bitcast2:]] = OpBitcast %[[#IntPtr]] %[[#Cast2]]
; CHECK: %[[#CmpXChg2:]] = OpAtomicCompareExchange %[[#Int]] %[[#Bitcast2]] %[[#Scope]] %[[#MemSem]] %[[#MemSem]] %[[#Select2]] %[[#Phi2]]
; CHECK: %[[#Eq3:]] = OpIEqual %[[#Bool]] %[[#CmpXChg2]] %[[#Phi2]]
; CHECK: %[[#Insert3:]] = OpCompositeInsert %[[#Struct]] %[[#CmpXChg2]] %[[#UndefStruct]] 0
; CHECK: %[[#Insert4:]] = OpCompositeInsert %[[#Struct]] %[[#Eq3]] %[[#Insert3]] 1
; CHECK: %[[#Cond2:]] = OpCompositeExtract %[[#Bool]] %[[#Insert4]] 1
; CHECK: %[[#Phi2Next]] = OpCompositeExtract %[[#Int]] %[[#Insert4]] 0
; CHECK: OpBranchConditional %[[#Cond2]] %[[#Exit2:]] %[[#Loop2]]
; CHECK: %[[#Exit2]] = OpLabel

  ret void
; CHECK: OpReturn
; CHECK: OpFunctionEnd
}
