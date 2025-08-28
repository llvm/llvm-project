; RUN: llc < %s -mtriple=nvptx64 -O2 -mcpu=sm_70 -mattr=+ptx63 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | %ptxas-verify -arch=sm_70 %}

declare void @llvm.nvvm.nanosleep(i32)

; CHECK-LABEL: test_nanosleep_r
define void @test_nanosleep_r(i32 noundef %d) {
entry:
; CHECK: nanosleep.u32   %[[REG:.+]];
  call void @llvm.nvvm.nanosleep(i32 %d)
  ret void
}

; CHECK-LABEL: test_nanosleep_i
define void @test_nanosleep_i() {
entry:
; CHECK: nanosleep.u32   42;
  call void @llvm.nvvm.nanosleep(i32 42)
  ret void
}
