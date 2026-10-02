; RUN: llc -mtriple=spirv64-unknown-unknown -O0 < %s | FileCheck %s
; RUN: llc -mtriple=spirv32-unknown-unknown -O0 < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown < %s -o - -filetype=obj | spirv-val %}

; SPIR-V does not have a stack or return address concept.
; llvm.returnaddress and llvm.frameaddress are lowered to null (OpConstantNull).

declare ptr @llvm.returnaddress(i32)
declare ptr @llvm.frameaddress(i32)

; CHECK-DAG: %[[#PTR_TY:]] = OpTypePointer
; CHECK-DAG: %[[#NULL:]] = OpConstantNull %[[#PTR_TY]]

; CHECK: %[[#]] = OpFunction
; CHECK: OpReturnValue %[[#NULL]]
define ptr @test_returnaddress() {
  %ret = call ptr @llvm.returnaddress(i32 0)
  ret ptr %ret
}

; CHECK: %[[#]] = OpFunction
; CHECK: OpReturnValue %[[#NULL]]
define ptr @test_frameaddress() {
  %ret = call ptr @llvm.frameaddress(i32 0)
  ret ptr %ret
}
