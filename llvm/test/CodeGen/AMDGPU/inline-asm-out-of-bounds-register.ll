; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=bonaire -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error %s

; CHECK: error: couldn't allocate output register for constraint '{v256}'
define void @out_of_bounds_vgpr32_def() {
  %v = tail call i32 asm sideeffect "v_mov_b32 $0, -1", "={v256}"()
  ret void
}

; CHECK: error: couldn't allocate output register for constraint '{v[255:256]}'
define void @out_of_bounds_vgpr64_def_high_tuple() {
  %v = tail call i32 asm sideeffect "v_mov_b32 $0, -1", "={v[255:256]}"()
  ret void
}

; CHECK: error: couldn't allocate output register for constraint '{v[256:257]}'
define void @out_of_bounds_vgpr64_def_low_tuple() {
  %v = tail call i32 asm sideeffect "v_mov_b32 $0, -1", "={v[256:257]}"()
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v256}'
define void @out_of_bounds_vgpr32_use() {
  %v = tail call i32 asm sideeffect "v_mov_b32 %0, %1", "=v,{v256}"(i32 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[255:256]}'
define void @out_of_bounds_vgpr64_high_tuple() {
  tail call void asm sideeffect "; use %0", "{v[255:256]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[256:257]}'
define void @out_of_bounds_vgpr64_low_tuple() {
  tail call void asm sideeffect "; use %0", "{v[256:257]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[1:0]}'
define void @vgpr_tuple_swapped() {
  tail call void asm sideeffect "; use %0", "{v[1:0]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v4294967295}'
define void @vgpr_uintmax() {
  tail call void asm sideeffect "; use %0", "{v4294967295}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v4294967296}'
define void @vgpr_uintmax_p1() {
  tail call void asm sideeffect "; use %0", "{v4294967296}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[4294967295:4294967296]}'
define void @vgpr_tuple_uintmax() {
  tail call void asm sideeffect "; use %0", "{v[4294967295:4294967296]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[0:4294967295]}'
define void @vgpr_tuple_0_uintmax() {
  tail call void asm sideeffect "; use %0", "{v[0:4294967295]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[0:4294967296]}'
define void @vgpr_tuple_0_uintmax_p1() {
  tail call void asm sideeffect "; use %0", "{v[0:4294967296]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[4294967264:4294967295]}'
define void @vgpr32_last_is_uintmax() {
  tail call void asm sideeffect "; use %0", "{v[4294967264:4294967295]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[4294967265:4294967296]}'
define void @vgpr32_last_is_uintmax_p1() {
  tail call void asm sideeffect "; use %0", "{v[4294967265:4294967296]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[2:2147483651]}'
define void @overflow_bitwidth_0() {
  tail call void asm sideeffect "; use %0", "{v[2:2147483651]}"(i64 123)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint '{v[2147483635:2147483651]}'
define void @overflow_bitwidth_1() {
  tail call void asm sideeffect "; use %0", "{v[2147483635:2147483651]}"(i64 123)
  ret void
}

