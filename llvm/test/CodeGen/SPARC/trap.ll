; RUN: llc -mtriple=sparc-linux-gnu < %s -show-mc-encoding | FileCheck %s

define void @test1() {
  tail call void @llvm.trap()
  unreachable

; CHECK-LABEL: test1:
; CHECK: ta 5 ! encoding: [0x91,0xd0,0x20,0x05]
}

declare void @llvm.trap()
