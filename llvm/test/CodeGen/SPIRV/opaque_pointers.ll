; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK

; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PtrInt32Ty:]] = OpTypePointer Function %[[#Int32Ty]]
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrInt64Ty:]] = OpTypePointer Function %[[#Int64Ty]]
; CHECK-DAG: %[[#FTy:]] = OpTypeFunction %[[#Int64Ty]] %[[#PtrInt32Ty]]
; CHECK-DAG: %[[#Const:]] = OpConstant %[[#Int32Ty]] 0
; CHECK: OpFunction %[[#Int64Ty]] None %[[#FTy]]
; CHECK: %[[#Param:]] = OpFunctionParameter %[[#PtrInt32Ty]]
; CHECK: OpStore %[[#Param]] %[[#Const]] Aligned 4
; CHECK-DAG: %[[#Bitcast:]] = OpBitcast %[[#PtrInt64Ty]] %[[#Param]]
; CHECK: %[[#Res:]] = OpLoad %[[#Int64Ty]] %[[#Bitcast]] Aligned 4
; CHECK: OpReturnValue %[[#Res]]

define i64 @test(ptr %p) {
  store i32 0, ptr %p, align 4
  %v = load i64, ptr %p, align 4
  ret i64 %v
}
