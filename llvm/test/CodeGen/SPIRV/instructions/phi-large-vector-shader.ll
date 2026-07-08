; RUN: llc -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; In Shader execution models the SPIR-V max vector size is 4, so a G_PHI on
; a wider vector must be split into multiple PHIs of width 4.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V4:]] = OpTypeVector %[[#I32]] 4
; CHECK-COUNT-8: %[[#PHI:]] = OpPhi %[[#V4]]
; CHECK: OpCompositeExtract %[[#I32]] %[[#PHI]]

@A = internal addrspace(10) global [8 x <4 x i32>] zeroinitializer
@Out = internal addrspace(10) global [8 x <4 x i32>] zeroinitializer
@Cond = internal addrspace(10) global i32 zeroinitializer

define void @main() local_unnamed_addr #0 {
entry:
  %c = load i32, ptr addrspace(10) @Cond
  %cond = icmp ne i32 %c, 0
  %p0 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @A, i32 0, i32 0
  %a0 = load <4 x i32>, ptr addrspace(10) %p0
  %p1 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @A, i32 0, i32 1
  %a1 = load <4 x i32>, ptr addrspace(10) %p1
  %ab = shufflevector <4 x i32> %a0, <4 x i32> %a1,
              <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %wide_a = shufflevector <8 x i32> %ab, <8 x i32> %ab,
              <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                          i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                          i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                          i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %wide_b = shufflevector <8 x i32> %ab, <8 x i32> %ab,
              <32 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                          i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                          i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                          i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %p = phi <32 x i32> [ %wide_a, %then ], [ %wide_b, %else ]
  %s0 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %s1 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %s2 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %s3 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %s4 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
  %s5 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>
  %s6 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>
  %s7 = shufflevector <32 x i32> %p, <32 x i32> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>
  %o0 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 0
  store <4 x i32> %s0, ptr addrspace(10) %o0
  %o1 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 1
  store <4 x i32> %s1, ptr addrspace(10) %o1
  %o2 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 2
  store <4 x i32> %s2, ptr addrspace(10) %o2
  %o3 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 3
  store <4 x i32> %s3, ptr addrspace(10) %o3
  %o4 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 4
  store <4 x i32> %s4, ptr addrspace(10) %o4
  %o5 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 5
  store <4 x i32> %s5, ptr addrspace(10) %o5
  %o6 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 6
  store <4 x i32> %s6, ptr addrspace(10) %o6
  %o7 = getelementptr [8 x <4 x i32>], ptr addrspace(10) @Out, i32 0, i32 7
  store <4 x i32> %s7, ptr addrspace(10) %o7
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
