; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s
; RUN: opt -passes='default<O2>' -S %s | FileCheck %s
;
; Target string attributes survive bitcode and optimization. Calls with identical
; arithmetic operands but different hard register classes must not be merged.
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)

define <4 x float> @mixed(float %a, float %b, <4 x float> %c) {
; CHECK-LABEL: define <4 x float> @mixed(
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32({{.*}}){{( #[0-9]+)?}}{{$}}
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32({{.*}}) #[[AGPR:[0-9]+]]
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32({{.*}}) #[[VGPR:[0-9]+]]
  %d0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  %d1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %d2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  %t = fadd <4 x float> %d0, %d1
  %sum = fadd <4 x float> %t, %d2
  ret <4 x float> %sum
}

; CHECK-DAG: attributes #[[AGPR]] = { {{.*}}"amdgpu-mfma-cd-regclass"="agpr"{{.*}} }
; CHECK-DAG: attributes #[[VGPR]] = { {{.*}}"amdgpu-mfma-cd-regclass"="vgpr"{{.*}} }
