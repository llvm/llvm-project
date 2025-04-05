; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown-opencl %s -o - | FileCheck %s

; CHECK: OpUnreachable
define void @test_unreachable() {
  unreachable
}
