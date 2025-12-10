; RUN: llc -O0 -mtriple=spirv32-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown < %s -filetype=obj | spirv-val %}
;
; Some OpenCL builtins have mixed vector-scalar variants, but OpExtInt only supports
; versions where all the arguments have the same type.
;
; We generate code, but it is invalid.
; We should generate vector versions for these cases.

define spir_kernel void @S_MIN() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function S_MIN
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} s_min %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x i32> @_Z3minDv2_ii(<2 x i32> <i32 1, i32 10>, i32 5)
  ret void
}

define spir_kernel void @U_MIN() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function U_MIN
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} u_min %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x i32> @_Z3minDv2_jj(<2 x i32> <i32 1, i32 10>, i32 5)
  ret void
}

define spir_kernel void @S_MAX() {
; CHECK-LABEL: OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function S_MAX
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} s_max %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x i32> @_Z3maxDv2_ii(<2 x i32> <i32 1, i32 10>, i32 5)
  ret void
}

define spir_kernel void @F_MIN() {
; CHECK-LABEL: OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function F_MIN
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} fmin %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x float> @_Z3minDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @F_MAX() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function F_MAX
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} fmax %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x float> @_Z3maxDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @F_FMIN() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function F_FMIN
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} fmin %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x float> @_Z4fminDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @F_FMAX() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function F_FMAX
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} fmax %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x float> @_Z4fmaxDv2_ff(<2 x float> <float 1.0, float 10.0>, float 5.0)
  ret void
}

define spir_kernel void @S_CLAMP() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function S_CLAMP
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC_0:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR_0:[0-9]+]] %[[SCALAR_0]]
; CHECK-NEXT:    %[[VEC_1:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR_1:[0-9]+]] %[[SCALAR_1]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} s_clamp %{{[0-9]+}} %[[VEC_0]] %[[VEC_1]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x i32> @_Z5clampDv2_iii(<2 x i32> <i32 1, i32 10>, i32 5, i32 6)
  ret void
}

define spir_kernel void @F_CLAMP() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function F_CLAMP
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC_0:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR_0:[0-9]+]] %[[SCALAR_0]]
; CHECK-NEXT:    %[[VEC_1:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR_1:[0-9]+]] %[[SCALAR_1]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} fclamp %{{[0-9]+}} %[[VEC_0]] %[[VEC_1]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x float> @_Z5clampDv2_fff(<2 x float> <float 1.0, float 10.0>, float 5.0, float 6.0)
  ret void
}

define spir_kernel void @MIX() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function MIX
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR:[0-9]+]] %[[SCALAR]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} mix %{{[0-9]+}} %{{[0-9]+}} %[[VEC]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x float> @_Z3mixDv2_fS_f(<2 x float> <float 1.0, float 10.0>, <2 x float> <float 2.0, float 20.0>, float 0.5)
  ret void
}

define spir_kernel void @SMOOTHSTEP() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function SMOOTHSTEP
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %[[VEC_0:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR_0:[0-9]+]] %[[SCALAR_0]]
; CHECK-NEXT:    %[[VEC_1:[0-9]+]] = OpCompositeConstruct %[[VECTYPE:[0-9]+]] %[[SCALAR_1:[0-9]+]] %[[SCALAR_1]]
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %[[VECTYPE]] %{{[0-9]+}} smoothstep %[[VEC_0]] %[[VEC_1]] %{{[0-9]+}}
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  %call = tail call spir_func <2 x float> @_Z10smoothstepffDv2_f(float 1.0, float 0.5, <2 x float> <float 1.0, float 10.0>)
  ret void
}

define spir_kernel void @ill_0() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function ill_0
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    OpFunctionCall %{{[0-9]+}} %{{[0-9]+}}
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  tail call spir_func void @_Z3minv()
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
