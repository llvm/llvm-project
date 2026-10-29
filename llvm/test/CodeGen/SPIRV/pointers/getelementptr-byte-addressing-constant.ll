; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - \
; RUN:   | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 \
; RUN:   -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj \
; RUN:   | spirv-val --target-env vulkan1.3 %}

; Check nested arrays, struct fields, vector elements, and value casts.
@Nested = external addrspace(3) global [2 x [4 x i32]]
@Struct = external addrspace(3) global { i32, [4 x i32] }
@Vectors = external addrspace(3) global [2 x <4 x i32>]
@Output = external addrspace(3) global float

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#INT_PTR:]] = OpTypePointer Workgroup %[[#INT]]

define void @main() #0 {
; CHECK: OpFunction
; CHECK: %[[#ARRAY:]] = OpInBoundsAccessChain
; CHECK: %[[#NESTED:]] = OpInBoundsAccessChain %[[#INT_PTR]] %[[#ARRAY]]
; CHECK: OpStore %[[#NESTED]]
; CHECK: %[[#FIELD:]] = OpInBoundsAccessChain %[[#INT_PTR]]
; CHECK: OpStore %[[#FIELD]]
; CHECK: %[[#ELEMENT:]] = OpInBoundsAccessChain %[[#INT_PTR]]
; CHECK: OpStore %[[#ELEMENT]]
; CHECK: %[[#LOADED:]] = OpLoad %[[#INT]] %[[#ELEMENT]]
; CHECK: %[[#CAST:]] = OpBitcast %[[#FLOAT]] %[[#LOADED]]
; CHECK: OpStore {{%[0-9]+}} %[[#CAST]]
entry:
  store i32 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Nested, i64 16), align 4
  store i32 43, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Struct, i64 8), align 4
  store i32 44, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Vectors, i64 20), align 4
  %value = load float, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Vectors, i64 20), align 4
  store float %value, ptr addrspace(3) @Output, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
