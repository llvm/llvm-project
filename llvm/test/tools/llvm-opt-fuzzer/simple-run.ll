; Check that fuzzer will succeed on a trivial input
; REQUIRES: x86-registered-target

; Temporary bitcode file
; RUN: opt -o %t %s

; RUN: llvm-opt-fuzzer %t -ignore_remaining_args=1 -mtriple x86_64 -passes instcombine 2>&1 | FileCheck %s
; CHECK: Running

define i32 @test(i32 %n) {
entry:
  ret i32 0
}
