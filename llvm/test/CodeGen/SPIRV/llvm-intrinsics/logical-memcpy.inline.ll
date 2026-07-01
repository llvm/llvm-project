; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck -implicit-check-not=OpFunctionCall %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Dst:]] "dst"
; CHECK-DAG: OpName %[[#Src:]] "src"

; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#StructS:]] = OpTypeStruct %[[#Float32]] %[[#Float32]] %[[#Float32]] %[[#Float32]] %[[#Float32]]
; CHECK-DAG: %[[#Ptr_CrossWG_StructS:]] = OpTypePointer CrossWorkgroup %[[#StructS]]
%struct.S = type <{ float, float, float, float, float }>

; CHECK-DAG: %[[#Src]] = OpVariable %[[#Ptr_CrossWG_StructS]] CrossWorkgroup
@src = external dso_local addrspace(1) global %struct.S, align 4

; CHECK-DAG: %[[#Dst]] = OpVariable %[[#Ptr_CrossWG_StructS]] CrossWorkgroup
@dst = external dso_local addrspace(1) global %struct.S, align 4

; CHECK: OpCopyMemory %[[#Dst]] %[[#Src]]

define void @main() local_unnamed_addr #0 {
  call void @llvm.memcpy.inline.p0.p0.i64(ptr addrspace(1) align 4 @dst, ptr addrspace(1) align 4 @src, i64 20, i1 false)
  ret void
}

declare void @llvm.memcpy.inline.p0.p0.i64(ptr addrspace(1) captures(none), ptr addrspace(1) captures(none) readonly, i64 immarg, i1 immarg)

attributes #0 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
