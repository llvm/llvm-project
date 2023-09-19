; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK

; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrTy:]] = OpTypePointer Function %[[#Int8Ty]]
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#FTy:]] = OpTypeFunction %[[#Int64Ty]] %[[#PtrTy]]
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const:]] = OpConstant %[[#Int32Ty]] 0
; CHECK: OpFunction %[[#Int64Ty]] None %[[#FTy]]
; CHECK: %[[#Parm:]] = OpFunctionParameter %[[#PtrTy]]
; CHECK: OpStore %[[#Parm]] %[[#Const]] Aligned 4
; CHECK: %[[#Res:]] = OpLoad %[[#Int64Ty]] %[[#Parm]] Aligned 8
; CHECK: OpReturnValue %[[#Res]]

define i64 @test(ptr %p) {
  store i32 0, ptr %p
  %v = load i64, ptr %p
  ret i64 %v
}
