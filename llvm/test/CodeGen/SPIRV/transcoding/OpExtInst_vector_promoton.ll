; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o -
;
; Some OpenCL builtins have mixed vector-scalar variants, but OpExtInt only supports
; versions where all the arguments have the same type.

define spir_kernel void @S_MIN() {
entry:
  %call = tail call spir_func <2 x i32> @_Z3minDv2_ii(<2 x i32> <i32 1, i32 10>, i32 5)
  ret void
}

define spir_kernel void @U_MIN() {
entry:
  %call = tail call spir_func <2 x i32> @_Z3minDv2_jj(<2 x i32> <i32 1, i32 10>, i32 5)
  ret void
}

define spir_kernel void @S_MAX() {
entry:
  %call = tail call spir_func <2 x i32> @_Z3maxDv2_ii(<2 x i32> <i32 1, i32 10>, i32 5)
  ret void
}

define spir_kernel void @F_MIN() {
entry:
  %call = tail call spir_func <2 x float> @_Z3minDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @F_MAX() {
entry:
  %call = tail call spir_func <2 x float> @_Z3maxDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @F_FMIN() {
entry:
  %call = tail call spir_func <2 x float> @_Z4fminDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @F_FMAX() {
entry:
  %call = tail call spir_func <2 x float> @_Z4fmaxDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @S_CLAMP() {
entry:
  %call = tail call spir_func <2 x i32> @_Z5clampDv2_iii(<2 x i32> <i32 1, i32 10>, i32 5, i32 6)
  ret void
}

define spir_kernel void @F_CLAMP() {
entry:
  %call = tail call spir_func <2 x float> @_Z5clampDv2_fff(<2 x float> <float 1.0, float 10.0>, float 5.0, float 6.0)
  ret void
}

define spir_kernel void @MIX() {
entry:
  %call = tail call spir_func <2 x float> @_Z3mixDv2_fS_f(<2 x float> <float 1.0, float 10.0>, <2 x float> <float 2.0, float 20.0>, float 0.5)
  ret void
}

define spir_kernel void @SMOOTHSTEP() {
entry:
  %call = tail call spir_func <2 x float> @_Z10smoothstepffDv2_f(float 1.0, float 0.5, <2 x float> <float 1.0, float 10.0>)
  ret void
}

define spir_kernel void @ill_0() {
entry:
  tail call spir_func void @_Z3minv()
  ret void
}

define spir_kernel void @ill_1() {
entry:
  tail call spir_func void @_Z3miniii(i32 1, i32 2, i32 3)
  ret void
}

declare spir_func <2 x i32> @_Z3minDv2_ii(<2 x i32>, i32)
declare spir_func <2 x i32> @_Z3minDv2_jj(<2 x i32>, i32)
declare spir_func <2 x i32> @_Z3maxDv2_ii(<2 x i32>, i32)
declare spir_func <2 x float> @_Z3minDv2_ff(<2 x float>, float)
declare spir_func <2 x float> @_Z3maxDv2_ff(<2 x float>, float)
declare spir_func <2 x float> @_Z4fminDv2_ff(<2 x float>, float)
declare spir_func <2 x float> @_Z4fmaxDv2_ff(<2 x float>, float)
declare spir_func <2 x i32> @_Z5clampDv2_iii(<2 x i32>, i32)
declare spir_func <2 x float> @_Z5clampDv2_fff(<2 x float>, float)
declare spir_func <2 x float> @_Z3mixDv2_fS_f(<2 x float>, <2 x float>, float)
declare spir_func <2 x float> @_Z10smoothstepffDv2_f(float, float, <2 x float>)
declare spir_func void @_Z3minv()
declare spir_func i32 @_Z3miniii(i32, i32, i32)
