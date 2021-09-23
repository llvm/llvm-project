; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GCN,GFX89 %s
; END.

; GCN-LABEL: 'insertelement_v2'
; GCN: estimated cost of 0 for {{.*}} insertelement <2 x i32>
; GCN: estimated cost of 0 for {{.*}} insertelement <2 x i64>
; CI: estimated cost of 1 for {{.*}} insertelement <2 x i16>
; GFX89: estimated cost of 0 for {{.*}} insertelement <2 x i16>
; GCN: estimated cost of 1 for {{.*}} insertelement <2 x i16>
; GCN: estimated cost of 1 for {{.*}} insertelement <2 x i8>
define amdgpu_kernel void @insertelement_v2() {
  %v2i32_1 = insertelement <2 x i32> undef, i32 123, i32 1
  %v2i64_1 = insertelement <2 x i64> undef, i64 123, i64 1
  %v2i16_0 = insertelement <2 x i16> undef, i16 123, i16 0
  %v2i16_1 = insertelement <2 x i16> undef, i16 123, i16 1
  %v2i8_1 = insertelement <2 x i8> undef, i8 123, i8 1
  ret void
}
