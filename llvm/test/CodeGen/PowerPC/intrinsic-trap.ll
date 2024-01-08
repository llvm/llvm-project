; REQUIRES: asserts
; RUN: not --crash llc -verify-machineinstrs -mtriple=powerpc64le-- < %s 2>&1 | FileCheck %s
; CHECK: Bad machine code: Non-terminator instruction after the first terminator

define i32 @test() {
  call void @llvm.trap()
  ret i32 0
}

declare void @llvm.trap()
