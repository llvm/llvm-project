; RUN: llc -mtriple=armv7 %s -o - | FileCheck %s

; CHECK: dmb ld

define void @test2() #0 {
  call void @llvm.arm.dmb(i32 13)
  ret void
}

declare void @llvm.arm.dmb(i32)

attributes #0 = { "target-cpu"="cyclone" }
