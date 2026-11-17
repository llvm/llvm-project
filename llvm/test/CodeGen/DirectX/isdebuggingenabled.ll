; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.10-library %s | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.10-compute %s | FileCheck %s
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s 2>&1 | FileCheck %s --check-prefix=ERROR

; CHECK-LABEL: define i1 @test(
; CHECK: %[[RESULT:.*]] = call i1 @dx.op.isDebuggingEnabled(i32 344)
; CHECK-NEXT: ret i1 %[[RESULT]]
; CHECK: declare i1 @dx.op.isDebuggingEnabled(i32)

; ERROR: Cannot create IsDebuggingEnabled operation: No valid stage for DXIL version 1.9

define i1 @test() {
entry:
  %result = call i1 @llvm.dx.isdebuggingenabled()
  ret i1 %result
}

declare i1 @llvm.dx.isdebuggingenabled()
