; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,FAST64,FAST16 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SLOW64,SLOW16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,FAST16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,SLOW16 %s
; END.

; ALL-LABEL: 'shl'
; ALL: estimated cost of 1 for {{.*}} shl i32
; FAST64: estimated cost of 2 for {{.*}} shl i64
; SLOW64: estimated cost of 4 for {{.*}} shl i64
; SIZEALL: estimated cost of 2 for {{.*}} shl i64
; ALL: estimated cost of 1 for {{.*}} shl i16
; SLOW16: estimated cost of 2 for {{.*}} shl <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} shl <2 x i16>
define amdgpu_kernel void @shl() #0 {
  %i32 = shl i32 undef, undef
  %i64 = shl i64 undef, undef
  %i16 = shl i16 undef, undef
  %v2i16 = shl <2 x i16> undef, undef
  ret void
}

; ALL-LABEL: 'lshr'
; ALL: estimated cost of 1 for {{.*}} lshr i32
; FAST64: estimated cost of 2 for {{.*}} lshr i64
; SLOW64: estimated cost of 4 for {{.*}} lshr i64
; SIZEALL: estimated cost of 2 for {{.*}} lshr i64
; ALL: estimated cost of 1 for {{.*}} lshr i16
; SLOW16: estimated cost of 2 for {{.*}} lshr <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} lshr <2 x i16>
define amdgpu_kernel void @lshr() #0 {
  %i32 = lshr i32 undef, undef
  %i64 = lshr i64 undef, undef
  %i16 = lshr i16 undef, undef
  %v2i16 = lshr <2 x i16> undef, undef
  ret void
}

; ALL-LABEL: 'ashr'
; ALL: estimated cost of 1 for {{.*}} ashr i32
; FAST64: estimated cost of 2 for {{.*}} ashr i64
; SLOW64: estimated cost of 4 for {{.*}} ashr i64
; ALL: estimated cost of 1 for {{.*}} ashr i16
; SLOW16: estimated cost of 2 for {{.*}} ashr <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} ashr <2 x i16>
define amdgpu_kernel void @ashr() #0 {
  %i32 = ashr i32 undef, undef
  %i64 = ashr i64 undef, undef
  %i16 = ashr i16 undef, undef
  %v2i16 = ashr <2 x i16> undef, undef
  ret void
}

attributes #0 = { nounwind }
