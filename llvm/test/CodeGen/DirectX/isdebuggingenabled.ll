; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.10-library %s | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.10-compute %s | FileCheck %s
; RUN: llc -O3 -mtriple=dxil-pc-shadermodel6.10-compute %s -o - | FileCheck %s
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s 2>&1 | FileCheck %s --check-prefix=ERROR

; CHECK-LABEL: define i1 @test(
; CHECK: %[[RESULT1:.*]] = call i1 @dx.op.isDebuggingEnabled(i32 344)
; CHECK: call void @dx.op.debugBreak(i32 343)
; CHECK: %[[RESULT2:.*]] = call i1 @dx.op.isDebuggingEnabled(i32 344)
; CHECK: %[[RETURN:.*]] = xor i1 %[[RESULT2]], %[[RESULT1]]
; CHECK-NEXT: ret i1 %[[RETURN]]
; CHECK: declare i1 @dx.op.isDebuggingEnabled(i32)

; ERROR: Cannot create IsDebuggingEnabled operation: No valid stage for DXIL version 1.9

define i1 @test() {
entry:
  %result2 = call i1 @llvm.dx.isdebuggingenabled()
  call void @llvm.debugtrap()
  %result1 = call i1 @llvm.dx.isdebuggingenabled()
  %ret = xor i1 %result1, %result2
  ret i1 %ret
}

declare i1 @llvm.dx.isdebuggingenabled()
