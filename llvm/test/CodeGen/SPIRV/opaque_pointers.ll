; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK

; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrInt8Ty:]] = OpTypePointer Function %[[#Int8Ty]]
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PtrInt32Ty:]] = OpTypePointer Function %[[#Int32Ty]]
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrInt64Ty:]] = OpTypePointer Function %[[#Int64Ty]]
; CHECK-DAG: %[[#FTy:]] = OpTypeFunction %[[#Int64Ty]] %[[#PtrInt8Ty]]
; CHECK-DAG: %[[#Const:]] = OpConstant %[[#Int32Ty]] 0
; CHECK: OpFunction %[[#Int64Ty]] None %[[#FTy]]
; CHECK: %[[#Parm:]] = OpFunctionParameter %[[#PtrInt8Ty]]
; CHECK-DAG: %[[#Bitcast1:]] = OpBitcast %[[#PtrInt32Ty]] %[[#Parm]]
; CHECK: OpStore %[[#Bitcast1]] %[[#Const]] Aligned 4
; CHECK-DAG: %[[#Bitcast2:]] = OpBitcast %[[#PtrInt64Ty]] %[[#Parm]]
; CHECK: %[[#Res:]] = OpLoad %[[#Int64Ty]] %[[#Bitcast2]] Aligned 4
; CHECK: OpReturnValue %[[#Res]]

define i64 @test(ptr %p) {
  store i32 0, ptr %p, align 4
  %v = load i64, ptr %p, align 4
  ret i64 %v
}
