; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK

; CHECK: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK: %[[#PtrTy:]] = OpTypePointer Function %[[#Int8Ty]]
; CHECK: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK: %[[#FTy:]] = OpTypeFunction %[[#Int64Ty]] %[[#PtrTy]]
; CHECK: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK: %[[#Const:]] = OpConstant %[[#Int32Ty]] 0
; CHECK: OpFunction %[[#Int64Ty]] None %[[#FTy]]
; CHECK: %[[#Parm:]] = OpFunctionParameter %[[#PtrTy]]
; CHECK: OpStore %[[#Parm]] %[[#Const]] Aligned 4
; CHECK: %[[#Res:]] = OpLoad %[[#Int64Ty]] %[[#Parm]] Aligned 8
; CHECK: OpReturnValue %[[#Res]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"

define i64 @test(ptr %p) {
  store i32 0, ptr %p
  %v = load i64, ptr %p
  ret i64 %v
}
