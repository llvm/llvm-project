; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Test that zext of boolean vectors with sizes that exceed valid SPIR-V
; OpTypeVector component counts (9 from 3x3, 12 from 3x4 bool matrix
; transpose) compile without errors in shader mode where MaxVectorSize is 4.
; The bools are grouped into legal-sized sub-vectors (<3 x i1> or <4 x i1>),
; zext is lowered to OpSelect, then the results are reassembled.
; See https://github.com/llvm/llvm-project/issues/186864

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#v9int:]] = OpTypeVector %[[#int]] 9
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#v3bool:]] = OpTypeVector %[[#bool]] 3
; CHECK-DAG: %[[#v3int:]] = OpTypeVector %[[#int]] 3
; CHECK-DAG: %[[#v12int:]] = OpTypeVector %[[#int]] 12
; CHECK-DAG: %[[#v4bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG: %[[#v4int:]] = OpTypeVector %[[#int]] 4
; CHECK-DAG: %[[#const_0:]] = OpConstant %[[#int]] 0
; CHECK-DAG: %[[#const_1:]] = OpConstant %[[#int]] 1
; CHECK-DAG: %[[#v3zeros:]] = OpConstantComposite %[[#v3int]] %[[#const_0]] %[[#const_0]] %[[#const_0]]
; CHECK-DAG: %[[#v3ones:]] = OpConstantComposite %[[#v3int]] %[[#const_1]] %[[#const_1]] %[[#const_1]]
; CHECK-DAG: %[[#v4zeros:]] = OpConstantComposite %[[#v4int]] %[[#const_0]] %[[#const_0]] %[[#const_0]] %[[#const_0]]
; CHECK-DAG: %[[#v4ones:]] = OpConstantComposite %[[#v4int]] %[[#const_1]] %[[#const_1]] %[[#const_1]] %[[#const_1]]

define internal void @test_zext_v9i1(ptr addrspace(10) %out, ptr addrspace(10) %in) {
; CHECK: OpFunction
entry:
; CHECK: %[[#grp0:]] = OpCompositeConstruct %[[#v3bool]]
; CHECK: %[[#grp1:]] = OpCompositeConstruct %[[#v3bool]]
; CHECK: %[[#grp2:]] = OpCompositeConstruct %[[#v3bool]]
; CHECK: %[[#sel0:]] = OpSelect %[[#v3int]] %[[#grp0]] %[[#v3ones]] %[[#v3zeros]]
; CHECK: %[[#sel1:]] = OpSelect %[[#v3int]] %[[#grp1]] %[[#v3ones]] %[[#v3zeros]]
; CHECK: %[[#sel2:]] = OpSelect %[[#v3int]] %[[#grp2]] %[[#v3ones]] %[[#v3zeros]]
; CHECK: %[[#res9:]] = OpCompositeConstruct %[[#v9int]]
; CHECK: OpStore %{{.+}} %[[#res9]]
  %src = load <9 x i1>, ptr addrspace(10) %in
  %ext = zext <9 x i1> %src to <9 x i32>
  store <9 x i32> %ext, ptr addrspace(10) %out
  ret void
}

define internal void @test_zext_v12i1(ptr addrspace(10) %out, ptr addrspace(10) %in) {
; CHECK: OpFunction
entry:
; CHECK: %[[#grp3:]] = OpCompositeConstruct %[[#v4bool]]
; CHECK: %[[#grp4:]] = OpCompositeConstruct %[[#v4bool]]
; CHECK: %[[#grp5:]] = OpCompositeConstruct %[[#v4bool]]
; CHECK: %[[#sel3:]] = OpSelect %[[#v4int]] %[[#grp3]] %[[#v4ones]] %[[#v4zeros]]
; CHECK: %[[#sel4:]] = OpSelect %[[#v4int]] %[[#grp4]] %[[#v4ones]] %[[#v4zeros]]
; CHECK: %[[#sel5:]] = OpSelect %[[#v4int]] %[[#grp5]] %[[#v4ones]] %[[#v4zeros]]
; CHECK: %[[#res12:]] = OpCompositeConstruct %[[#v12int]]
; CHECK: OpStore %{{.+}} %[[#res12]]
  %src = load <12 x i1>, ptr addrspace(10) %in
  %ext = zext <12 x i1> %src to <12 x i32>
  store <12 x i32> %ext, ptr addrspace(10) %out
  ret void
}

define void @main() local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
