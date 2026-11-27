; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Regression test for https://github.com/llvm/llvm-project/issues/206815.
;
; InstCombine rewrites `color.data[o + 2]` as `color.data[o]` plus 32 bytes.
; A logical access chain cannot step from one array element into the next, so
; the backend must turn those offsets into an array index. In this test `color`
; is a `struct { float4 data[8]; }`, so each element is 16 bytes.
;
;   color.data[o] + 16 bytes  ->  color.data[o + 1]
;   color.data[o] + 36 bytes  ->  color.data[o + 2].y
;   color.data[o] - 12 bytes  ->  color.data[o - 1].y
;   color.data[o] + 20 bytes  ->  color.data[o + 1].y  (inbounds is kept)
;   color.data[o] +  4 bytes  ->  color.data[o].y      (left unchanged)

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#idx_2:]] = OpConstant %[[#idx:]] 2{{$}}
; CHECK-DAG: %[[#idx_1:]] = OpConstant %[[#idx]] 1{{$}}
; CHECK-DAG: %[[#idx_m1:]] = OpConstant %[[#idx]] {{4294967295|18446744073709551615}}{{$}}
; CHECK: %[[#base:]] = OpAccessChain %[[#]] %[[#]] %[[#]]{{$}}
; CHECK: %[[#o:]] = OpLoad %[[#uint]]
; CHECK: %[[#e0:]] = OpInBoundsAccessChain %[[#]] %[[#base]] %[[#zero:]] %[[#o]]{{$}}
; CHECK: %[[#o1:]] = OpIAdd %[[#idx]] %[[#]] %[[#idx_1]]
; CHECK: OpAccessChain %[[#]] %[[#base]] %[[#zero]] %[[#o1]]{{$}}
; CHECK: %[[#o2:]] = OpIAdd %[[#idx]] %[[#]] %[[#idx_2]]
; CHECK: %[[#e2:]] = OpAccessChain %[[#]] %[[#base]] %[[#zero]] %[[#o2]]{{$}}
; CHECK: OpAccessChain %[[#]] %[[#e2]] %[[#y:]]{{$}}
; CHECK: %[[#om1:]] = OpIAdd %[[#idx]] %[[#]] %[[#idx_m1]]
; CHECK: %[[#em1:]] = OpAccessChain %[[#]] %[[#base]] %[[#zero]] %[[#om1]]{{$}}
; CHECK: OpAccessChain %[[#]] %[[#em1]] %[[#y]]{{$}}
; CHECK: %[[#o3:]] = OpIAdd %[[#idx]] %[[#]] %[[#idx_1]]
; CHECK: %[[#e3:]] = OpInBoundsAccessChain %[[#]] %[[#base]] %[[#zero]] %[[#o3]]{{$}}
; CHECK: OpInBoundsAccessChain %[[#]] %[[#e3]] %[[#y]]{{$}}
; CHECK-NOT: OpIAdd
; CHECK: OpAccessChain %[[#]] %[[#e0]] %[[#y]]{{$}}

%Col = type <{ [8 x <4 x float>] }>

@.str = private unnamed_addr constant [6 x i8] c"color\00", align 1
@Offset = external hidden thread_local local_unnamed_addr addrspace(7) externally_initialized constant i32, !spirv.Decorations !2
@Out = external hidden thread_local local_unnamed_addr addrspace(8) global <4 x float>, !spirv.Decorations !3

define void @main() #0 {
entry:
  %h = tail call target("spirv.VulkanBuffer", %Col, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Cols_2_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str)
  %base = call ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Cols_2_0t(target("spirv.VulkanBuffer", %Col, 2, 0) %h)
  %o = load i32, ptr addrspace(7) @Offset, align 4
  %p0 = getelementptr inbounds nuw [16 x i8], ptr addrspace(12) %base, i32 %o
  %v0 = load <4 x float>, ptr addrspace(12) %p0, align 4
  %p1 = getelementptr i8, ptr addrspace(12) %p0, i32 16
  %v1 = load <4 x float>, ptr addrspace(12) %p1, align 4
  %p2 = getelementptr i8, ptr addrspace(12) %p0, i32 36
  %f2 = load float, ptr addrspace(12) %p2, align 4
  %pm1 = getelementptr i8, ptr addrspace(12) %p0, i32 -12
  %fm1 = load float, ptr addrspace(12) %pm1, align 4
  %p3 = getelementptr inbounds i8, ptr addrspace(12) %p0, i32 20
  %f3 = load float, ptr addrspace(12) %p3, align 4
  %p4 = getelementptr i8, ptr addrspace(12) %p0, i32 4
  %f4 = load float, ptr addrspace(12) %p4, align 4
  %s = fadd <4 x float> %v0, %v1
  %e = extractelement <4 x float> %s, i64 0
  %r2 = fadd float %e, %f2
  %r3 = fadd float %r2, %fm1
  %r4 = fadd float %r3, %f3
  %r = fadd float %r4, %f4
  %v = insertelement <4 x float> %s, float %r, i64 0
  store <4 x float> %v, ptr addrspace(8) @Out, align 4
  ret void
}

declare target("spirv.VulkanBuffer", %Col, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Cols_2_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Cols_2_0t(target("spirv.VulkanBuffer", %Col, 2, 0))

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{i32 30, i32 0}
!1 = !{i32 14}
!2 = !{!0, !1}
!3 = !{!0}
