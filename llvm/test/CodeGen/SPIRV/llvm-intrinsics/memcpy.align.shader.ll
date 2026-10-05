; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --implicit-check-not=Aligned --implicit-check-not=OpCopyMemorySized
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Shader copies omit alignment operands, including when the source and
; destination alignments differ. Volatile still applies to both accesses.
; CHECK: OpCapability Shader
; CHECK: OpMemoryModel Logical GLSL450
; CHECK: OpEntryPoint GLCompute %[[#Main:]] "main"
; CHECK-DAG: OpName %[[#Dst:]] "dst"
; CHECK-DAG: OpName %[[#Src:]] "src"

%struct.S = type { i32, i32 }
@dst = internal addrspace(3) global %struct.S poison, align 8
@src = internal addrspace(3) global %struct.S poison, align 8

; CHECK: %[[#Main]] = OpFunction
define void @main() #0 {
entry:
; CHECK: OpCopyMemory %[[#Dst]] %[[#Src]]{{$}}
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 4 @dst, ptr addrspace(3) align 4 @src, i32 8, i1 false)
; CHECK: OpCopyMemory %[[#Dst]] %[[#Src]]{{$}}
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 8 @dst, ptr addrspace(3) align 4 @src, i32 8, i1 false)
; CHECK: OpCopyMemory %[[#Dst]] %[[#Src]]{{$}}
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 4 @dst, ptr addrspace(3) align 8 @src, i32 8, i1 false)
; CHECK: OpCopyMemory %[[#Dst]] %[[#Src]]{{$}}
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 4 @dst, ptr addrspace(3) @src, i32 8, i1 false)
; CHECK: OpCopyMemory %[[#Dst]] %[[#Src]] Volatile{{$}}
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 8 @dst, ptr addrspace(3) align 4 @src, i32 8, i1 true)
  ret void
; CHECK: OpReturn
; CHECK: OpFunctionEnd
}

declare void @llvm.memcpy.p3.p3.i32(ptr addrspace(3), ptr addrspace(3), i32, i1 immarg)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
