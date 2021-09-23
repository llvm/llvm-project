; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=ALL,SLOW16 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL,FAST16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=ALL,SLOW16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL,FAST16 %s
; END.

; ALL-LABEL: 'or'
; ALL: estimated cost of 1 for {{.*}} or i32
; ALL: estimated cost of 2 for {{.*}} or i64
; SLOW16: estimated cost of 2 for {{.*}} or <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} or <2 x i16>
define amdgpu_kernel void @or() #0 {
  %i32 = or i32 undef, undef
  %i64 = or i64 undef, undef
  %v2i16 = or <2 x i16> undef, undef
  ret void
}

; ALL-LABEL: 'xor'
; ALL: estimated cost of 1 for {{.*}} xor i32
; ALL: estimated cost of 2 for {{.*}} xor i64
; SLOW16: estimated cost of 2 for {{.*}} xor <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} xor <2 x i16>
define amdgpu_kernel void @xor() #0 {
  %i32 = xor i32 undef, undef
  %i64 = xor i64 undef, undef
  %v2i16 = xor <2 x i16> undef, undef
  ret void
}

; ALL-LABEL: 'and'
; ALL: estimated cost of 1 for {{.*}} and i32
; ALL: estimated cost of 2 for {{.*}} and i64
; SLOW16: estimated cost of 2 for {{.*}} and <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} and <2 x i16>
define amdgpu_kernel void @and() #0 {
  %i32 = and i32 undef, undef
  %i64 = and i64 undef, undef
  %v2i16 = and <2 x i16> undef, undef
  ret void
}

attributes #0 = { nounwind }
