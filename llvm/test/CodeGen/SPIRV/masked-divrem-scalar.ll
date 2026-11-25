; RUN: llc -mtriple=spirv64 -O0 < %s | FileCheck %s

; CHECK-DAG:  %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG:  %[[#BOOL:]] = OpTypeBool
; CHECK-DAG:  %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG:  %[[#ONE64:]] = OpConstant %[[#I64]] 1
; CHECK-DAG:  %[[#ONE32:]] = OpConstant %[[#I32]] 1
; CHECK:      OpFunction
; CHECK-NEXT: %[[#P32:]] = OpFunctionParameter
; CHECK-NEXT: %[[#P64:]] = OpFunctionParameter
; CHECK-NEXT: %[[#A32:]] = OpFunctionParameter %[[#I32]]
; CHECK-NEXT: %[[#B32:]] = OpFunctionParameter %[[#I32]]
; CHECK-NEXT: %[[#A64:]] = OpFunctionParameter %[[#I64]]
; CHECK-NEXT: %[[#B64:]] = OpFunctionParameter %[[#I64]]
; CHECK-NEXT: %[[#M:]] = OpFunctionParameter %[[#BOOL]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#D1:]] = OpSelect %[[#I32]] %[[#M]] %[[#B32]] %[[#ONE32]]
; CHECK-NEXT: %[[#]] = OpUDiv %[[#I32]] %[[#A32]] %[[#D1]]
; CHECK-NEXT: %[[#D2:]] = OpSelect %[[#I32]] %[[#M]] %[[#B32]] %[[#ONE32]]
; CHECK-NEXT: %[[#]] = OpSDiv %[[#I32]] %[[#A32]] %[[#D2]]
; CHECK-NEXT: %[[#D3:]] = OpSelect %[[#I32]] %[[#M]] %[[#B32]] %[[#ONE32]]
; CHECK-NEXT: %[[#]] = OpUMod %[[#I32]] %[[#A32]] %[[#D3]]
; CHECK-NEXT: %[[#D4:]] = OpSelect %[[#I32]] %[[#M]] %[[#B32]] %[[#ONE32]]
; CHECK-NEXT: %[[#]] = OpSRem %[[#I32]] %[[#A32]] %[[#D4]]
; CHECK:      %[[#D5:]] = OpSelect %[[#I64]] %[[#M]] %[[#B64]] %[[#ONE64]]
; CHECK-NEXT: %[[#]] = OpUDiv %[[#I64]] %[[#A64]] %[[#D5]]
; CHECK-NEXT: %[[#D6:]] = OpSelect %[[#I64]] %[[#M]] %[[#B64]] %[[#ONE64]]
; CHECK-NEXT: %[[#]] = OpSDiv %[[#I64]] %[[#A64]] %[[#D6]]
; CHECK-NEXT: %[[#D7:]] = OpSelect %[[#I64]] %[[#M]] %[[#B64]] %[[#ONE64]]
; CHECK-NEXT: %[[#]] = OpUMod %[[#I64]] %[[#A64]] %[[#D7]]
; CHECK-NEXT: %[[#D8:]] = OpSelect %[[#I64]] %[[#M]] %[[#B64]] %[[#ONE64]]
; CHECK-NEXT: %[[#]] = OpSRem %[[#I64]] %[[#A64]] %[[#D8]]
; CHECK:      OpFunctionEnd

define void @masked_divrem(ptr %0, ptr %1, i32 %2, i32 %3, i64 %4, i64 %5, i1 %6) {
  %8 = call i32 @llvm.masked.udiv.i32(i32 %2, i32 %3, i1 %6)
  %9 = call i32 @llvm.masked.sdiv.i32(i32 %2, i32 %3, i1 %6)
  %10 = call i32 @llvm.masked.urem.i32(i32 %2, i32 %3, i1 %6)
  %11 = call i32 @llvm.masked.srem.i32(i32 %2, i32 %3, i1 %6)
  store i32 %8, ptr %0
  store i32 %9, ptr %0
  store i32 %10, ptr %0
  store i32 %11, ptr %0
  %12 = call i64 @llvm.masked.udiv.i64(i64 %4, i64 %5, i1 %6)
  %13 = call i64 @llvm.masked.sdiv.i64(i64 %4, i64 %5, i1 %6)
  %14 = call i64 @llvm.masked.urem.i64(i64 %4, i64 %5, i1 %6)
  %15 = call i64 @llvm.masked.srem.i64(i64 %4, i64 %5, i1 %6)
  store i64 %12, ptr %1
  store i64 %13, ptr %1
  store i64 %14, ptr %1
  store i64 %15, ptr %1
  ret void
}
