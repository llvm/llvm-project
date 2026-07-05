; REQUIRES: asserts
; RUN: not --crash llc -verify-machineinstrs -mtriple=powerpc64-unknown-unknown \
; RUN:   < %s 2>&1 | FileCheck %s

declare void @llvm.ppc.sync()
declare void @llvm.ppc.cfence.i128(i128)

define void @test_cfence(i128 %src) {
entry:
  call void @llvm.ppc.sync()
; CHECK: ExpandIntegerOperand Op{{.*}}llvm.ppc.cfence
; CHECK: LLVM ERROR: Do not know how to expand this operator's operand!
  call void @llvm.ppc.cfence.i128(i128 %src)
  ret void
}
