; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=SLOW16,THRPTALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=FAST16,THRPTALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=SIZESLOW16,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=SIZEFAST16,SIZEALL,ALL %s
; END.

; ALL-LABEL: 'mul_i32'
; THRPTALL: estimated cost of 4 for {{.*}} mul i32
; SIZEALL: estimated cost of 2 for {{.*}} mul i32
; THRPTALL: estimated cost of 8 for {{.*}} mul <2 x i32>
; SIZEALL: estimated cost of 4 for {{.*}} mul <2 x i32>
; THRPTALL: estimated cost of 12 for {{.*}} mul <3 x i32>
; SIZEALL: estimated cost of 6 for {{.*}} mul <3 x i32>
; THRPTALL: estimated cost of 16 for {{.*}} mul <4 x i32>
; SIZEALL: estimated cost of 8 for {{.*}} mul <4 x i32>
; THRPTALL: estimated cost of 20 for {{.*}} mul <5 x i32>
; SIZEALL: estimated cost of 10 for {{.*}} mul <5 x i32>
define amdgpu_kernel void @mul_i32() #0 {
  %i32 = mul i32 undef, undef
  %v2i32 = mul <2 x i32> undef, undef
  %v3i32 = mul <3 x i32> undef, undef
  %v4i32 = mul <4 x i32> undef, undef
  %v5i32 = mul <5 x i32> undef, undef
  ret void
}

; ALL-LABEL: 'mul_i64'
; THRPTALL: estimated cost of 20 for {{.*}} mul i64
; SIZEALL: estimated cost of 12 for {{.*}} mul i64
; THRPTALL: estimated cost of 40 for {{.*}} mul <2 x i64>
; SIZEALL: estimated cost of 24 for {{.*}} mul <2 x i64>
; THRPTALL: estimated cost of 60 for {{.*}} mul <3 x i64>
; SIZEALL: estimated cost of 36 for {{.*}} mul <3 x i64>
; THRPTALL: estimated cost of 80 for {{.*}} mul <4 x i64>
; SIZEALL: estimated cost of 48 for {{.*}} mul <4 x i64>
; THRPTALL: estimated cost of 320 for {{.*}} mul <8 x i64>
; SIZEALL: estimated cost of 192 for {{.*}} mul <8 x i64>
define amdgpu_kernel void @mul_i64() #0 {
  %i64 = mul i64 undef, undef
  %v2i64 = mul <2 x i64> undef, undef
  %v3i64 = mul <3 x i64> undef, undef
  %v4i64 = mul <4 x i64> undef, undef
  %v8i64 = mul <8 x i64> undef, undef
  ret void
}

; ALL-LABEL: 'mul_i16'
; THRPTALL: estimated cost of 4 for {{.*}} mul i16
; SIZEALL: estimated cost of 2 for {{.*}} mul i16
; SLOW16: estimated cost of 8 for {{.*}} mul <2 x i16>
; FAST16: estimated cost of 4 for {{.*}} mul <2 x i16>
; SIZESLOW16: estimated cost of 4 for {{.*}} mul <2 x i16>
; SIZEFAST16: estimated cost of 2 for {{.*}} mul <2 x i16>
; SLOW16: estimated cost of 16 for {{.*}} mul <3 x i16>
; FAST16: estimated cost of 8 for {{.*}} mul <3 x i16>
; SIZESLOW16: estimated cost of 8 for {{.*}} mul <3 x i16>
; SIZEFAST16: estimated cost of 4 for {{.*}} mul <3 x i16>
define amdgpu_kernel void @mul_i16() #0 {
  %i16 = mul i16 undef, undef
  %v2i16 = mul <2 x i16> undef, undef
  %v3i16 = mul <3 x i16> undef, undef
  ret void
}

attributes #0 = { nounwind }
