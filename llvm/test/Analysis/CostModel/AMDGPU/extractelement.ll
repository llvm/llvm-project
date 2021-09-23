; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GCN,GFX89 %s
; END.

; GCN-LABEL: 'extractelement_32'
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <2 x i32>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <2 x float>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <3 x i32>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <4 x i32>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <5 x i32>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <8 x i32>
; GCN-NEXT: estimated cost of 2 for {{.*}} extractelement <8 x i32>
define amdgpu_kernel void @extractelement_32(i32 %arg) {
  %v2i32_1 = extractelement <2 x i32> undef, i32 1
  %v2f32_1 = extractelement <2 x float> undef, i32 1
  %v3i32_1 = extractelement <3 x i32> undef, i32 1
  %v4i32_1 = extractelement <4 x i32> undef, i32 1
  %v5i32_1 = extractelement <5 x i32> undef, i32 1
  %v8i32_1 = extractelement <8 x i32> undef, i32 1
  %v8i32_a = extractelement <8 x i32> undef, i32 %arg
  ret void
}

; GCN-LABEL: 'extractelement_64'
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <2 x i64>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <3 x i64>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <4 x i64>
; GCN-NEXT: estimated cost of 0 for {{.*}} extractelement <8 x i64>
define amdgpu_kernel void @extractelement_64() {
  %v2i64_1 = extractelement <2 x i64> undef, i64 1
  %v3i64_1 = extractelement <3 x i64> undef, i64 1
  %v4i64_1 = extractelement <4 x i64> undef, i64 1
  %v8i64_1 = extractelement <8 x i64> undef, i64 1
  ret void
}

; GCN-LABEL: 'extractelement_8'
; GCN-NEXT: estimated cost of 1 for {{.*}} extractelement <4 x i8>
define amdgpu_kernel void @extractelement_8() {
  %v4i8_1 = extractelement <4 x i8> undef, i8 1
  ret void
}

; GCN-LABEL: 'extractelement_16'
; CI-NEXT: estimated cost of 1 for {{.*}} extractelement <2 x i16> undef, i16 0
; GFX89-NEXT: estimated cost of 0 for {{.*}} extractelement <2 x i16>
; GCN-NEXT: estimated cost of 1 for {{.*}} extractelement <2 x i16>
; GCN-NEXT: estimated cost of 1 for {{.*}} extractelement <2 x i16>
define amdgpu_kernel void @extractelement_16(i32 %arg) {
  %v2i16_0 = extractelement <2 x i16> undef, i16 0
  %v2i16_1 = extractelement <2 x i16> undef, i16 1
  %v2i16_a = extractelement <2 x i16> undef, i32 %arg
  ret void
}
