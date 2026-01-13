; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#v4int:]] = OpTypeVector %[[#int]] 4
; CHECK-DAG: %[[#v4double:]] = OpTypeVector %[[#double]] 4
; CHECK-DAG: %[[#v2int:]] = OpTypeVector %[[#int]] 2
; CHECK-DAG: %[[#v2double:]] = OpTypeVector %[[#double]] 2
; CHECK-DAG: %[[#v3int:]] = OpTypeVector %[[#int]] 3
; CHECK-DAG: %[[#v3double:]] = OpTypeVector %[[#double]] 3
; CHECK-DAG: %[[#ptr_v4double:]] = OpTypePointer Private %[[#v4double]]
; CHECK-DAG: %[[#ptr_v4int:]] = OpTypePointer Private %[[#v4int]]
; CHECK-DAG: %[[#ptr_v3double:]] = OpTypePointer Private %[[#v3double]]
; CHECK-DAG: %[[#ptr_v3int:]] = OpTypePointer Private %[[#v3int]]
; CHECK-DAG: %[[#GVec4:]] = OpVariable %[[#ptr_v4double]] Private
; CHECK-DAG: %[[#Lows:]] = OpVariable %[[#ptr_v4int]] Private
; CHECK-DAG: %[[#Highs:]] = OpVariable %[[#ptr_v4int]] Private
; CHECK-DAG: %[[#GVec3:]] = OpVariable %[[#ptr_v3double]] Private
; CHECK-DAG: %[[#Lows3:]] = OpVariable %[[#ptr_v3int]] Private
; CHECK-DAG: %[[#Highs3:]] = OpVariable %[[#ptr_v3int]] Private

@GVec4 = internal addrspace(10) global <4 x double> zeroinitializer
@Lows = internal addrspace(10) global <4 x i32> zeroinitializer
@Highs = internal addrspace(10) global <4 x i32> zeroinitializer
@GVec3 = internal addrspace(10) global <3 x double> zeroinitializer
@Lows3 = internal addrspace(10) global <3 x i32> zeroinitializer
@Highs3 = internal addrspace(10) global <3 x i32> zeroinitializer

; Test splitting a vector of size 8.
define internal void @test_split() {
entry:
  ; CHECK: %[[#load_v4double:]] = OpLoad %[[#v4double]] %[[#GVec4]]
  ; CHECK: %[[#v2double_01:]] = OpVectorShuffle %[[#v2double]] %[[#load_v4double]] %{{[a-zA-Z0-9_]+}} 0 1
  ; CHECK: %[[#v2double_23:]] = OpVectorShuffle %[[#v2double]] %[[#load_v4double]] %{{[a-zA-Z0-9_]+}} 2 3
  ; CHECK: %[[#v4int_01:]] = OpBitcast %[[#v4int]] %[[#v2double_01]]
  ; CHECK: %[[#v4int_23:]] = OpBitcast %[[#v4int]] %[[#v2double_23]]
  %0 = load <8 x i32>, ptr addrspace(10) @GVec4, align 32

  ; CHECK: %[[#l0:]] = OpCompositeExtract %[[#int]] %[[#v4int_01]] 0
  ; CHECK: %[[#l1:]] = OpCompositeExtract %[[#int]] %[[#v4int_01]] 2
  ; CHECK: %[[#l2:]] = OpCompositeExtract %[[#int]] %[[#v4int_23]] 0
  ; CHECK: %[[#l3:]] = OpCompositeExtract %[[#int]] %[[#v4int_23]] 2
  ; CHECK: %[[#res_low:]] = OpCompositeConstruct %[[#v4int]] %[[#l0]] %[[#l1]] %[[#l2]] %[[#l3]]
  %1 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  
  ; CHECK: %[[#h0:]] = OpCompositeExtract %[[#int]] %[[#v4int_01]] 1
  ; CHECK: %[[#h1:]] = OpCompositeExtract %[[#int]] %[[#v4int_01]] 3
  ; CHECK: %[[#h2:]] = OpCompositeExtract %[[#int]] %[[#v4int_23]] 1
  ; CHECK: %[[#h3:]] = OpCompositeExtract %[[#int]] %[[#v4int_23]] 3
  ; CHECK: %[[#res_high:]] = OpCompositeConstruct %[[#v4int]] %[[#h0]] %[[#h1]] %[[#h2]] %[[#h3]]
  %2 = shufflevector <8 x i32> %0, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x i32> %1, ptr addrspace(10) @Lows, align 16
  store <4 x i32> %2, ptr addrspace(10) @Highs, align 16
  ret void
}

define internal void @test_recombine() {
entry:
  ; CHECK: %[[#l:]] = OpLoad %[[#v4int]] %[[#Lows]]
  %0 = load <4 x i32>, ptr addrspace(10) @Lows, align 16
  ; CHECK: %[[#h:]] = OpLoad %[[#v4int]] %[[#Highs]]
  %1 = load <4 x i32>, ptr addrspace(10) @Highs, align 16

  ; CHECK-DAG: %[[#l0:]] = OpCompositeExtract %[[#int]] %[[#l]] 0
  ; CHECK-DAG: %[[#l1:]] = OpCompositeExtract %[[#int]] %[[#l]] 1
  ; CHECK-DAG: %[[#l2:]] = OpCompositeExtract %[[#int]] %[[#l]] 2
  ; CHECK-DAG: %[[#l3:]] = OpCompositeExtract %[[#int]] %[[#l]] 3
  ; CHECK-DAG: %[[#h0:]] = OpCompositeExtract %[[#int]] %[[#h]] 0
  ; CHECK-DAG: %[[#h1:]] = OpCompositeExtract %[[#int]] %[[#h]] 1
  ; CHECK-DAG: %[[#h2:]] = OpCompositeExtract %[[#int]] %[[#h]] 2
  ; CHECK-DAG: %[[#h3:]] = OpCompositeExtract %[[#int]] %[[#h]] 3
  ; CHECK-DAG: %[[#v2i0:]] = OpCompositeConstruct %[[#v2int]] %[[#l0]] %[[#h0]]
  ; CHECK-DAG: %[[#d0:]] = OpBitcast %[[#double]] %[[#v2i0]]
  ; CHECK-DAG: %[[#v2i1:]] = OpCompositeConstruct %[[#v2int]] %[[#l1]] %[[#h1]]
  ; CHECK-DAG: %[[#d1:]] = OpBitcast %[[#double]] %[[#v2i1]]
  ; CHECK-DAG: %[[#v2i2:]] = OpCompositeConstruct %[[#v2int]] %[[#l2]] %[[#h2]]
  ; CHECK-DAG: %[[#d2:]] = OpBitcast %[[#double]] %[[#v2i2]]
  ; CHECK-DAG: %[[#v2i3:]] = OpCompositeConstruct %[[#v2int]] %[[#l3]] %[[#h3]]
  ; CHECK-DAG: %[[#d3:]] = OpBitcast %[[#double]] %[[#v2i3]]
  ; CHECK-DAG: %[[#res:]] = OpCompositeConstruct %[[#v4double]] %[[#d0]] %[[#d1]] %[[#d2]] %[[#d3]]
  %2 = shufflevector <4 x i32> %0, <4 x i32> %1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>

  ; CHECK: OpStore %[[#GVec4]] %[[#res]]
  store <8 x i32> %2, ptr addrspace(10) @GVec4, align 32
  ret void
}

; Test splitting a vector of size 6. It must be expanded to 8, and then split.
define internal void @test_bitcast_expand() {
entry:
  ; CHECK: %[[#load:]] = OpLoad %[[#v3double]] %[[#GVec3]]
  %0 = load <3 x double>, ptr addrspace(10) @GVec3, align 32

  ; CHECK: %[[#d0:]] = OpCompositeExtract %[[#double]] %[[#load]] 0
  ; CHECK: %[[#d1:]] = OpCompositeExtract %[[#double]] %[[#load]] 1
  ; CHECK: %[[#d2:]] = OpCompositeExtract %[[#double]] %[[#load]] 2
  ; CHECK: %[[#v2d0:]] = OpCompositeConstruct %[[#v2double]] %[[#d0]] %[[#d1]]
  ; CHECK: %[[#v2d1:]] = OpCompositeConstruct %[[#v2double]] %[[#d2]] %[[#]]
  ; CHECK: %[[#v4i0:]] = OpBitcast %[[#v4int]] %[[#v2d0]]
  ; CHECK: %[[#v4i1:]] = OpBitcast %[[#v4int]] %[[#v2d1]]
  %1 = bitcast <3 x double> %0 to <6 x i32>
  
  ; CHECK: %[[#l0:]] = OpCompositeExtract %[[#int]] %[[#v4i0]] 0
  ; CHECK: %[[#l1:]] = OpCompositeExtract %[[#int]] %[[#v4i0]] 2
  ; CHECK: %[[#l2:]] = OpCompositeExtract %[[#int]] %[[#v4i1]] 0
  ; CHECK: %[[#res_low:]] = OpCompositeConstruct %[[#v3int]] %[[#l0]] %[[#l1]] %[[#l2]]
  %2 = shufflevector <6 x i32> %1, <6 x i32> poison, <3 x i32> <i32 0, i32 2, i32 4>
  
  ; CHECK: %[[#h0:]] = OpCompositeExtract %[[#int]] %[[#v4i0]] 1
  ; CHECK: %[[#h1:]] = OpCompositeExtract %[[#int]] %[[#v4i0]] 3
  ; CHECK: %[[#h2:]] = OpCompositeExtract %[[#int]] %[[#v4i1]] 1
  ; CHECK: %[[#res_high:]] = OpCompositeConstruct %[[#v3int]] %[[#h0]] %[[#h1]] %[[#h2]]
  %3 = shufflevector <6 x i32> %1, <6 x i32> poison, <3 x i32> <i32 1, i32 3, i32 5>

  ; CHECK: OpStore %[[#Lows3]] %[[#res_low]]
  store <3 x i32> %2, ptr addrspace(10) @Lows3, align 16

  ; CHECK: OpStore %[[#Highs3]] %[[#res_high]]
  store <3 x i32> %3, ptr addrspace(10) @Highs3, align 16
  ret void
}

define void @main() local_unnamed_addr #0 {
entry:
  call void @test_split()
  call void @test_recombine()
  call void @test_bitcast_expand()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
