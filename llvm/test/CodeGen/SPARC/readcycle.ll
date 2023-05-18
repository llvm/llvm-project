; RUN: llc < %s -march=sparc -mcpu=gr740 -verify-machineinstrs | FileCheck %s
; CHECK: rd %asr23, %o1
; CHECK: mov %g0, %o0

define i64 @test() {
entry:
  %0 = call i64 @llvm.readcyclecounter()
  ret i64 %0
}

declare i64 @llvm.readcyclecounter()
