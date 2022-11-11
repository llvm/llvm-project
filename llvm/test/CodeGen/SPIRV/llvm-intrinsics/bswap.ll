; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpName %[[#FuncNameInt16:]] "spirv.llvm_bswap_i16"
; CHECK-SPIRV: OpName %[[#FuncNameInt32:]] "spirv.llvm_bswap_i32"
; CHECK-SPIRV: OpName %[[#FuncNameInt64:]] "spirv.llvm_bswap_i64"

; CHECK-SPIRV: %[[#TypeInt32:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#TypeInt16:]] = OpTypeInt 16 0
; CHECK-SPIRV: %[[#TypeInt64:]] = OpTypeInt 64 0

; CHECK-SPIRV: %[[#FuncNameInt16]] = OpFunction %[[#TypeInt16]]
; CHECK-SPIRV: %[[#FuncParameter:]] = OpFunctionParameter %[[#TypeInt16]]
; CHECK-SPIRV: %[[#]] = OpShiftLeftLogical %[[#TypeInt16]] %[[#FuncParameter]]
; CHECK-SPIRV: %[[#]] = OpShiftRightLogical %[[#TypeInt16]] %[[#FuncParameter]]
; CHECK-SPIRV: %[[#RetVal:]] = OpBitwiseOr %[[#TypeInt16]]
; CHECK-SPIRV: OpReturnValue %[[#RetVal]]
; CHECK-SPIRV: OpFunctionEnd

; CHECK-SPIRV: %[[#FuncNameInt32]] = OpFunction %[[#TypeInt32]]
; CHECK-SPIRV: %[[#FuncParameter:]] = OpFunctionParameter %[[#TypeInt32]]
; CHECK-SPIRV-COUNT-2: %[[#]] = OpShiftLeftLogical %[[#TypeInt32]] %[[#FuncParameter]]
; CHECK-SPIRV-COUNT-2: %[[#]] = OpShiftRightLogical %[[#TypeInt32]] %[[#FuncParameter]]
; CHECK-SPIRV-COUNT-2: OpBitwiseAnd %[[#TypeInt32]]
; CHECK-SPIRV-COUNT-2: OpBitwiseOr %[[#TypeInt32]]
; CHECK-SPIRV: %[[#RetVal:]] = OpBitwiseOr %[[#TypeInt32]]
; CHECK-SPIRV: OpReturnValue %[[#RetVal:]]
; CHECK-SPIRV: OpFunctionEnd

; CHECK-SPIRV: %[[#FuncNameInt64]]  = OpFunction %[[#TypeInt64]]
; CHECK-SPIRV: %[[#FuncParameter:]]  = OpFunctionParameter %[[#TypeInt64]]
; CHECK-SPIRV-COUNT-4: %[[#]] = OpShiftLeftLogical %[[#TypeInt64]] %[[#FuncParameter]] %[[#]]
; CHECK-SPIRV-COUNT-4: %[[#]] = OpShiftRightLogical %[[#TypeInt64]] %[[#FuncParameter]] %[[#]]
; CHECK-SPIRV-COUNT-6: OpBitwiseAnd %[[#TypeInt64]]
; CHECK-SPIRV-COUNT-6: OpBitwiseOr %[[#TypeInt64]]
; CHECK-SPIRV: %[[#RetVal:]] = OpBitwiseOr %[[#TypeInt64]]
; CHECK-SPIRV: OpReturnValue %[[#RetVal]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local i32 @main() {
entry:
  %retval = alloca i32, align 4
  %a = alloca i16, align 2
  %b = alloca i16, align 2
  %h = alloca i16, align 2
  %i = alloca i16, align 2
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i64, align 8
  %f = alloca i64, align 8
  store i32 0, i32* %retval, align 4
  store i16 258, i16* %a, align 2
  %0 = load i16, i16* %a, align 2
  %1 = call i16 @llvm.bswap.i16(i16 %0)
  store i16 %1, i16* %b, align 2
  store i16 234, i16* %h, align 2
  %2 = load i16, i16* %h, align 2
  %3 = call i16 @llvm.bswap.i16(i16 %2)
  store i16 %3, i16* %i, align 2
  store i32 566, i32* %c, align 4
  %4 = load i32, i32* %c, align 4
  %5 = call i32 @llvm.bswap.i32(i32 %4)
  store i32 %5, i32* %d, align 4
  store i64 12587, i64* %e, align 8
  %6 = load i64, i64* %e, align 8
  %7 = call i64 @llvm.bswap.i64(i64 %6)
  store i64 %7, i64* %f, align 8
  ret i32 0
}

declare i16 @llvm.bswap.i16(i16)

declare i32 @llvm.bswap.i32(i32)

declare i64 @llvm.bswap.i64(i64)
