; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A G_PHI on a vector wider than the SPIR-V max (16) must be split into
; multiple PHIs of the largest legal width.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V16:]] = OpTypeVector %[[#I32]] 16
; CHECK: %[[#PHI_LO:]] = OpPhi %[[#V16]]
; CHECK: %[[#PHI_HI:]] = OpPhi %[[#V16]]
; CHECK: OpCompositeExtract %[[#I32]] %[[#PHI_HI]]
; CHECK: OpIAdd %[[#V16]] %[[#PHI_LO]]

define spir_kernel void @phi_v32(ptr addrspace(1) %out, i1 %cond,
                                 <16 x i32> %a, <16 x i32> %b) {
entry:
  %wide_a = shufflevector <16 x i32> %a, <16 x i32> %b,
              <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                          i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                          i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                          i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %wide_b = shufflevector <16 x i32> %b, <16 x i32> %a,
              <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                          i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                          i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                          i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %p = phi <32 x i32> [ %wide_a, %then ], [ %wide_b, %else ]
  %sum = call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %p)
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.vector.reduce.add.v32i32(<32 x i32>)
