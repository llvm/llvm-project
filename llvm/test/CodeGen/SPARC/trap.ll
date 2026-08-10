; RUN: llc -mtriple=sparc-linux-gnu < %s -show-mc-encoding -verify-machineinstrs | FileCheck %s

define void @test1() {
  tail call void @llvm.trap()
  unreachable

; CHECK-LABEL: test1:
; CHECK: ta 5 ! encoding: [0x91,0xd0,0x20,0x05]
}

declare void @llvm.trap()

; CHECK-LABEL: testdebugtrap:
; CHECK: ta 1 ! encoding: [0x91,0xd0,0x20,0x01]
define void @testdebugtrap() {
entry:
  call void @llvm.debugtrap()
  ret void
}

declare void @llvm.debugtrap()
