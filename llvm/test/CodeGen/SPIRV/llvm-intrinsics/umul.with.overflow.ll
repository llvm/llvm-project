; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpName %[[#NAME_UMUL_FUNC_8:]] "spirv.llvm_umul_with_overflow_i8"
; CHECK-SPIRV: OpName %[[#NAME_UMUL_FUNC_32:]] "spirv.llvm_umul_with_overflow_i32"
; CHECK-SPIRV: OpName %[[#NAME_UMUL_FUNC_VEC_I64:]] "spirv.llvm_umul_with_overflow_v2i64"

define dso_local spir_func void @_Z4foo8hhPh(i8 zeroext %a, i8 zeroext %b, i8* nocapture %c) local_unnamed_addr {
entry:
  ; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#]] %[[#NAME_UMUL_FUNC_8]]
  %umul = tail call { i8, i1 } @llvm.umul.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %umul, 1
  %umul.value = extractvalue { i8, i1 } %umul, 0
  %storemerge = select i1 %cmp, i8 0, i8 %umul.value
  store i8 %storemerge, i8* %c, align 1
  ret void
}

define dso_local spir_func void @_Z5foo32jjPj(i32 %a, i32 %b, i32* nocapture %c) local_unnamed_addr {
entry:
  ; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#]] %[[#NAME_UMUL_FUNC_32]]
  %umul = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %b, i32 %a)
  %umul.val = extractvalue { i32, i1 } %umul, 0
  %umul.ov = extractvalue { i32, i1 } %umul, 1
  %spec.select = select i1 %umul.ov, i32 0, i32 %umul.val
  store i32 %spec.select, i32* %c, align 4
  ret void
}

define dso_local spir_func void @umulo_v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64>* %p) nounwind {
  ; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#]] %[[#NAME_UMUL_FUNC_VEC_I64]]
  %umul = call {<2 x i64>, <2 x i1>} @llvm.umul.with.overflow.v2i64(<2 x i64> %a, <2 x i64> %b)
  %umul.val = extractvalue {<2 x i64>, <2 x i1>} %umul, 0
  %umul.ov = extractvalue {<2 x i64>, <2 x i1>} %umul, 1
  %zero = alloca <2 x i64>, align 16
  %spec.select = select <2 x i1> %umul.ov, <2 x i64> <i64 0, i64 0>, <2 x i64> %umul.val
  store <2 x i64> %spec.select, <2 x i64>* %p
  ret void
}

; CHECK-SPIRV: %[[#NAME_UMUL_FUNC_8]] = OpFunction %[[#]]
; CHECK-SPIRV: %[[#VAR_A:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#VAR_B:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#MUL_RES:]] = OpIMul %[[#]] %[[#VAR_A]] %[[#VAR_B]]
; CHECK-SPIRV: %[[#DIV_RES:]] = OpUDiv %[[#]] %[[#MUL_RES]] %[[#VAR_A]]
; CHECK-SPIRV: %[[#CMP_RES:]] = OpINotEqual %[[#]] %[[#VAR_A]] %[[#DIV_RES]]
; CHECK-SPIRV: %[[#INSERT_RES:]] = OpCompositeInsert %[[#]] %[[#MUL_RES]]
; CHECK-SPIRV: %[[#INSERT_RES_1:]] = OpCompositeInsert %[[#]] %[[#CMP_RES]] %[[#INSERT_RES]]
; CHECK-SPIRV: OpReturnValue %[[#INSERT_RES_1]]

declare { i8, i1 } @llvm.umul.with.overflow.i8(i8, i8)

declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)

declare {<2 x i64>, <2 x i1>} @llvm.umul.with.overflow.v2i64(<2 x i64>, <2 x i64>)
