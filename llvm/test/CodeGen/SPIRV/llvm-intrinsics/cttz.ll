; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#topbit24:]] = OpConstant %[[#i32]] 16777216
; CHECK-DAG: %[[#i64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#topbit40:]] = OpConstant %[[#i64]] 1099511627776

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#extinst_id]] ctz
; CHECK: OpFunctionEnd

define spir_func i32 @TestCtz(i32 %x) local_unnamed_addr {
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  ret i32 %0
}

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#extinst_id]] ctz
; CHECK: OpFunctionEnd

define spir_func <4 x i32> @TestCtzVec(<4 x i32> %x) local_unnamed_addr {
entry:
  %0 = tail call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %x, i1 true)
  ret <4 x i32> %0
}

; Setting bit 24 keeps an all-zero i24 input counting 24 trailing zeros.
; CHECK: %[[#]] = OpFunction
; CHECK: %[[#orred24:]] = OpBitwiseOr %[[#i32]] %[[#]] %[[#topbit24]]
; CHECK: %[[#count24:]] = OpExtInst %[[#i32]] %[[#extinst_id]] ctz %[[#orred24]]
; CHECK: OpReturnValue %[[#count24]]

define spir_func i24 @TestCtzI24(i32 %x) {
  %narrow = trunc i32 %x to i24
  %count = call i24 @llvm.cttz.i24(i24 %narrow, i1 false)
  ret i24 %count
}

; The zero-poison form needs no fixup: its low bits are known non-zero.
; CHECK: %[[#]] = OpFunction
; CHECK-NOT: OpBitwiseOr
; CHECK: %[[#poison24:]] = OpExtInst %[[#i32]] %[[#extinst_id]] ctz %[[#]]
; CHECK: OpReturnValue %[[#poison24]]

define spir_func i24 @TestCtzI24ZeroPoison(i32 %x) {
  %narrow = trunc i32 %x to i24
  %count = call i24 @llvm.cttz.i24(i24 %narrow, i1 true)
  ret i24 %count
}

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#orred40:]] = OpBitwiseOr %[[#i64]] %[[#]] %[[#topbit40]]
; CHECK: %[[#count40:]] = OpExtInst %[[#i64]] %[[#extinst_id]] ctz %[[#orred40]]
; CHECK: OpReturnValue %[[#count40]]

define spir_func i40 @TestCtzI40(i40 %x) {
  %count = call i40 @llvm.cttz.i40(i40 %x, i1 false)
  ret i40 %count
}

declare i24 @llvm.cttz.i24(i24, i1 immarg)

declare i40 @llvm.cttz.i40(i40, i1 immarg)

declare i32 @llvm.cttz.i32(i32, i1 immarg)

declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1 immarg)
