; RUN: not llc < %s -mtriple nvptx 2>&1 | FileCheck %s

declare ptr @llvm.stackaddress.p0()

define ptr @test() {
; CHECK: error: <unknown>:0:0: STACKADDRESS is not supported on this target.
  %sp = call ptr @llvm.stackaddress.p0()
  ret ptr %sp
}
