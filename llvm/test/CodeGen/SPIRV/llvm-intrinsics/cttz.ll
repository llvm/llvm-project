; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#extinst_id]] ctz
; CHECK: OpFunctionEnd

define spir_func i32 @TestCtz(i32 %x) local_unnamed_addr {
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  ret i32 %0
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#extinst_id]] ctz
; CHECK: OpFunctionEnd

define spir_func <4 x i32> @TestCtzVec(<4 x i32> %x) local_unnamed_addr {
entry:
  %0 = tail call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %x, i1 true)
  ret <4 x i32> %0
}

declare i32 @llvm.cttz.i32(i32, i1 immarg)

declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1 immarg)
