; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FAST16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOW16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FAST16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOW16,ALL %s
; END.

; ALL-LABEL: 'add_i32'
; ALL: estimated cost of 1 for {{.*}} add i32
; ALL: estimated cost of 2 for {{.*}} add <2 x i32>
;;; Allow for 4 when v3i32 is illegal and TargetLowering thinks it needs widening,
;;; and 3 when it is legal.
; ALL: estimated cost of {{[34]}} for {{.*}} add <3 x i32>
; ALL: estimated cost of 4 for {{.*}} add <4 x i32>
;;; Allow for 8 when v3i32 is illegal and TargetLowering thinks it needs widening,
;;; and 5 when it is legal.
; ALL: estimated cost of {{[58]}} for {{.*}} add <5 x i32>
define amdgpu_kernel void @add_i32() #0 {
  %i32 = add i32 undef, undef
  %v2i32 = add <2 x i32> undef, undef
  %v3i32 = add <3 x i32> undef, undef
  %v4i32 = add <4 x i32> undef, undef
  %v5i32 = add <5 x i32> undef, undef
  ret void
}

; ALL-LABEL: 'add_i64'
; ALL: estimated cost of 2 for {{.*}} add i64
; ALL: estimated cost of 4 for {{.*}} add <2 x i64>
; ALL: estimated cost of 6 for {{.*}} add <3 x i64>
; ALL: estimated cost of 8 for {{.*}} add <4 x i64>
; ALL: estimated cost of 128 for {{.*}} add <16 x i64>
define amdgpu_kernel void @add_i64() #0 {
  %i64 = add i64 undef, undef
  %v2i64 = add <2 x i64> undef, undef
  %v3i64 = add <3 x i64> undef, undef
  %v4i64 = add <4 x i64> undef, undef
  %v16i64 = add <16 x i64> undef, undef
  ret void
}

; ALL-LABEL: 'add_i16'
; ALL: estimated cost of 1 for {{.*}} add i16
; SLOW16: estimated cost of 2 for {{.*}} add <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} add <2 x i16>
define amdgpu_kernel void @add_i16() #0 {
  %i16 = add i16 undef, undef
  %v2i16 = add <2 x i16> undef, undef
  ret void
}

; ALL-LABEL: 'sub'
; ALL: estimated cost of 1 for {{.*}} sub i32
; ALL: estimated cost of 2 for {{.*}} sub i64
; ALL: estimated cost of 1 for {{.*}} sub i16
; SLOW16: estimated cost of 2 for {{.*}} sub <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} sub <2 x i16>
define amdgpu_kernel void @sub() #0 {
  %i32 = sub i32 undef, undef
  %i64 = sub i64 undef, undef
  %i16 = sub i16 undef, undef
  %v2i16 = sub <2 x i16> undef, undef
  ret void
}

attributes #0 = { nounwind }
