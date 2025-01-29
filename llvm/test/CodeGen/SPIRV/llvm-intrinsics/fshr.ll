; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpName %[[#NAME_FSHR_FUNC_32:]] "spirv.llvm_fshr_i32"
; CHECK-SPIRV-DAG: OpName %[[#NAME_FSHR_FUNC_16:]] "spirv.llvm_fshr_i16"
; CHECK-SPIRV-DAG: OpName %[[#NAME_FSHR_FUNC_VEC_INT_16:]] "spirv.llvm_fshr_v2i16"
; CHECK-SPIRV-DAG: %[[#TYPE_INT_32:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#TYPE_ORIG_FUNC_32:]] = OpTypeFunction %[[#TYPE_INT_32]] %[[#TYPE_INT_32]] %[[#TYPE_INT_32]]
; CHECK-SPIRV-DAG: %[[#TYPE_INT_16:]] = OpTypeInt 16 0
; CHECK-SPIRV-DAG: %[[#TYPE_ORIG_FUNC_16:]] = OpTypeFunction %[[#TYPE_INT_16]] %[[#TYPE_INT_16]] %[[#TYPE_INT_16]]
; CHECK-SPIRV-DAG: %[[#TYPE_VEC_INT_16:]] = OpTypeVector %[[#TYPE_INT_16]] 2
; CHECK-SPIRV-DAG: %[[#TYPE_ORIG_FUNC_VEC_INT_16:]] = OpTypeFunction %[[#TYPE_VEC_INT_16]] %[[#TYPE_VEC_INT_16]] %[[#TYPE_VEC_INT_16]]
; CHECK-SPIRV-DAG: %[[#TYPE_FSHR_FUNC_32:]] = OpTypeFunction %[[#TYPE_INT_32]] %[[#TYPE_INT_32]] %[[#TYPE_INT_32]] %[[#TYPE_INT_32]]
; CHECK-SPIRV-DAG: %[[#TYPE_FSHR_FUNC_16:]] = OpTypeFunction %[[#TYPE_INT_16]] %[[#TYPE_INT_16]] %[[#TYPE_INT_16]] %[[#TYPE_INT_16]]
; CHECK-SPIRV-DAG: %[[#TYPE_FSHR_FUNC_VEC_INT_16:]] = OpTypeFunction %[[#TYPE_VEC_INT_16]] %[[#TYPE_VEC_INT_16]] %[[#TYPE_VEC_INT_16]] %[[#TYPE_VEC_INT_16]]
; CHECK-SPIRV-DAG: %[[#CONST_ROTATE_32:]] = OpConstant %[[#TYPE_INT_32]] 8
; CHECK-SPIRV-DAG: %[[#CONST_ROTATE_16:]] = OpConstant %[[#TYPE_INT_16]] 8
; CHECK-SPIRV-DAG: %[[#CONST_ROTATE_VEC_INT_16:]] = OpConstantComposite %[[#TYPE_VEC_INT_16]] %[[#CONST_ROTATE_16]] %[[#CONST_ROTATE_16]]
; CHECK-SPIRV-DAG: %[[#CONST_TYPE_SIZE_32:]] = OpConstant %[[#TYPE_INT_32]] 32

; CHECK-SPIRV: %[[#]] = OpFunction %[[#TYPE_INT_32]] {{.*}} %[[#TYPE_ORIG_FUNC_32]]
; CHECK-SPIRV: %[[#X:]] = OpFunctionParameter %[[#TYPE_INT_32]]
; CHECK-SPIRV: %[[#Y:]] = OpFunctionParameter %[[#TYPE_INT_32]]
define spir_func i32 @Test_i32(i32 %x, i32 %y) local_unnamed_addr {
entry:
  ; CHECK-SPIRV: %[[#CALL_32_X_Y:]] = OpFunctionCall %[[#TYPE_INT_32]] %[[#NAME_FSHR_FUNC_32]] %[[#X]] %[[#Y]] %[[#CONST_ROTATE_32]]
  %0 = call i32 @llvm.fshr.i32(i32 %x, i32 %y, i32 8)
  ; CHECK-SPIRV: %[[#CALL_32_Y_X:]] = OpFunctionCall %[[#TYPE_INT_32]] %[[#NAME_FSHR_FUNC_32]] %[[#Y]] %[[#X]] %[[#CONST_ROTATE_32]]
  %1 = call i32 @llvm.fshr.i32(i32 %y, i32 %x, i32 8)
  ; CHECK-SPIRV: %[[#ADD_32:]] = OpIAdd %[[#TYPE_INT_32]] %[[#CALL_32_X_Y]] %[[#CALL_32_Y_X]]
  %sum = add i32 %0, %1
  ; CHECK-SPIRV: OpReturnValue %[[#ADD_32]]
  ret i32 %sum
}

; CHECK-SPIRV: %[[#]] = OpFunction %[[#TYPE_INT_16]] {{.*}} %[[#TYPE_ORIG_FUNC_16]]
; CHECK-SPIRV: %[[#X:]] = OpFunctionParameter %[[#TYPE_INT_16]]
; CHECK-SPIRV: %[[#Y:]] = OpFunctionParameter %[[#TYPE_INT_16]]
define spir_func i16 @Test_i16(i16 %x, i16 %y) local_unnamed_addr {
entry:
  ; CHECK-SPIRV: %[[#CALL_16:]] = OpFunctionCall %[[#TYPE_INT_16]] %[[#NAME_FSHR_FUNC_16]] %[[#X]] %[[#Y]] %[[#CONST_ROTATE_16]]
  %0 = call i16 @llvm.fshr.i16(i16 %x, i16 %y, i16 8)
  ; CHECK-SPIRV: OpReturnValue %[[#CALL_16]]
  ret i16 %0
}

; CHECK-SPIRV: %[[#]] = OpFunction %[[#TYPE_VEC_INT_16]] {{.*}} %[[#TYPE_ORIG_FUNC_VEC_INT_16]]
; CHECK-SPIRV: %[[#X:]] = OpFunctionParameter %[[#TYPE_VEC_INT_16]]
; CHECK-SPIRV: %[[#Y:]] = OpFunctionParameter %[[#TYPE_VEC_INT_16]]
define spir_func <2 x i16> @Test_v2i16(<2 x i16> %x, <2 x i16> %y) local_unnamed_addr {
entry:
  ; CHECK-SPIRV: %[[#CALL_VEC_INT_16:]] = OpFunctionCall %[[#TYPE_VEC_INT_16]] %[[#NAME_FSHR_FUNC_VEC_INT_16]] %[[#X]] %[[#Y]] %[[#CONST_ROTATE_VEC_INT_16]]
  %0 = call <2 x i16> @llvm.fshr.v2i16(<2 x i16> %x, <2 x i16> %y, <2 x i16> <i16 8, i16 8>)
  ; CHECK-SPIRV: OpReturnValue %[[#CALL_VEC_INT_16]]
  ret <2 x i16> %0
}

; CHECK-SPIRV: %[[#NAME_FSHR_FUNC_32]] = OpFunction %[[#TYPE_INT_32]] {{.*}} %[[#TYPE_FSHR_FUNC_32]]
; CHECK-SPIRV: %[[#X_ARG:]] = OpFunctionParameter %[[#TYPE_INT_32]]
; CHECK-SPIRV: %[[#Y_ARG:]] = OpFunctionParameter %[[#TYPE_INT_32]]
; CHECK-SPIRV: %[[#ROT:]] = OpFunctionParameter %[[#TYPE_INT_32]]

; CHECK-SPIRV: %[[#ROTATE_MOD_SIZE:]] = OpUMod %[[#TYPE_INT_32]] %[[#ROT]] %[[#CONST_TYPE_SIZE_32]]
; CHECK-SPIRV: %[[#Y_SHIFT_RIGHT:]] = OpShiftRightLogical %[[#TYPE_INT_32]] %[[#Y_ARG]] %[[#ROTATE_MOD_SIZE]]
; CHECK-SPIRV: %[[#NEG_ROTATE:]] = OpISub %[[#TYPE_INT_32]] %[[#CONST_TYPE_SIZE_32]] %[[#ROTATE_MOD_SIZE]]
; CHECK-SPIRV: %[[#X_SHIFT_LEFT:]] = OpShiftLeftLogical %[[#TYPE_INT_32]] %[[#X_ARG]] %[[#NEG_ROTATE]]
; CHECK-SPIRV: %[[#FSHR_RESULT:]] = OpBitwiseOr %[[#TYPE_INT_32]] %[[#Y_SHIFT_RIGHT]] %[[#X_SHIFT_LEFT]]
; CHECK-SPIRV: OpReturnValue %[[#FSHR_RESULT]]

;; Just check that the function for i16 was generated as such - we've checked the logic for another type.
; CHECK-SPIRV: %[[#NAME_FSHR_FUNC_16]] = OpFunction %[[#TYPE_INT_16]] {{.*}} %[[#TYPE_FSHR_FUNC_16]]
; CHECK-SPIRV: %[[#X_ARG:]] = OpFunctionParameter %[[#TYPE_INT_16]]
; CHECK-SPIRV: %[[#Y_ARG:]] = OpFunctionParameter %[[#TYPE_INT_16]]
; CHECK-SPIRV: %[[#ROT:]] = OpFunctionParameter %[[#TYPE_INT_16]]

;; Just check that the function for v2i16 was generated as such - we've checked the logic for another type.
; CHECK-SPIRV: %[[#NAME_FSHR_FUNC_VEC_INT_16]] = OpFunction %[[#TYPE_VEC_INT_16]] {{.*}} %[[#TYPE_FSHR_FUNC_VEC_INT_16]]
; CHECK-SPIRV: %[[#X_ARG:]] = OpFunctionParameter %[[#TYPE_VEC_INT_16]]
; CHECK-SPIRV: %[[#Y_ARG:]] = OpFunctionParameter %[[#TYPE_VEC_INT_16]]
; CHECK-SPIRV: %[[#ROT:]] = OpFunctionParameter %[[#TYPE_VEC_INT_16]]

declare i32 @llvm.fshr.i32(i32, i32, i32)

declare i16 @llvm.fshr.i16(i16, i16, i16)

declare <2 x i16> @llvm.fshr.v2i16(<2 x i16>, <2 x i16>, <2 x i16>)
