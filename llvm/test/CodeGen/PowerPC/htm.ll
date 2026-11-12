; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+htm < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define zeroext i32 @test1() {
entry:
  %0 = tail call i32 @llvm.ppc.tbegin(i32 0)
  ret i32 %0

; CHECK-LABEL: @test1
; CHECK: tbegin. 0
; CHECK: mfocrf  [[REGISTER1:[0-9]+]], 128
; CHECK: rlwinm  [[REGISTER2:[0-9]+]], [[REGISTER1]], 3, 31, 31
; CHECK: xori    {{[0-9]+}}, [[REGISTER2]], 1
}

declare i32 @llvm.ppc.tbegin(i32) #1


define zeroext i32 @test2() {
entry:
  %0 = tail call i32 @llvm.ppc.tend(i32 0)
  ret i32 %0
; CHECK-LABEL: @test2
; CHECK: tend.
; CHECK: mfocrf  {{[0-9]+}}, 128
}

declare i32 @llvm.ppc.tend(i32)


define void @test3() {
entry:
  %0 = tail call i32 @llvm.ppc.tabort(i32 0)
  %1 = tail call i32 @llvm.ppc.tabortdc(i32 0, i32 1, i32 2)
  %2 = tail call i32 @llvm.ppc.tabortdci(i32 0, i32 1, i32 2)
  %3 = tail call i32 @llvm.ppc.tabortwc(i32 0, i32 1, i32 2)
  %4 = tail call i32 @llvm.ppc.tabortwci(i32 0, i32 1, i32 2)
  ret void
; CHECK-LABEL: @test3
; CHECK: tabort.    {{[0-9]+}}
; CHECK: tabortdc.  0, {{[0-9]+}}, {{[0-9]+}}
; CHECK: tabortdci. 0, {{[0-9]+}}, 2
; CHECK: tabortwc.  0, {{[0-9]+}}, {{[0-9]+}}
; CHECK: tabortwci. 0, {{[0-9]+}}, 2
}

declare i32 @llvm.ppc.tabort(i32)
declare i32 @llvm.ppc.tabortdc(i32, i32, i32)
declare i32 @llvm.ppc.tabortdci(i32, i32, i32)
declare i32 @llvm.ppc.tabortwc(i32, i32, i32)
declare i32 @llvm.ppc.tabortwci(i32, i32, i32)


define void @test4() {
entry:
  %0 = tail call i32 @llvm.ppc.tendall()
  %1 = tail call i32 @llvm.ppc.tresume()
  %2 = tail call i32 @llvm.ppc.tsuspend()
  %3 = tail call i64 @llvm.ppc.ttest()
  ret void
; CHECK-LABEL: @test4
; CHECK: tendall.
; CHECK: tresume.
; CHECK: tsuspend.
; CHECK: tabortwci. 0, {{[0-9]+}}, 0
}

declare i32 @llvm.ppc.tendall()
declare i32 @llvm.ppc.tresume()
declare i32 @llvm.ppc.tsuspend()
declare i64 @llvm.ppc.ttest()


define void @test5(i64 %v) {
entry:
  tail call void @llvm.ppc.set.texasr(i64 %v)
  tail call void @llvm.ppc.set.texasru(i64 %v)
  tail call void @llvm.ppc.set.tfhar(i64 %v)
  tail call void @llvm.ppc.set.tfiar(i64 %v)
  ret void
; CHECK-LABEL: @test5
; CHECK: mtspr 130, [[REG1:[0-9]+]]
; CHECK: mtspr 131, [[REG2:[0-9]+]]
; CHECK: mtspr 128, [[REG3:[0-9]+]]
; CHECK: mtspr 129, [[REG4:[0-9]+]]
}

define i64 @test6() {
entry:
  %0 = tail call i64 @llvm.ppc.get.texasr()
  ret i64 %0
; CHECK-LABEL: @test6
; CHECK: mfspr [[REG1:[0-9]+]], 130
}

define i64 @test7() {
entry:
  %0 = tail call i64 @llvm.ppc.get.texasru()
  ret i64 %0
; CHECK-LABEL: @test7
; CHECK: mfspr [[REG1:[0-9]+]], 131
}

define i64 @test8() {
entry:
  %0 = tail call i64 @llvm.ppc.get.tfhar()
  ret i64 %0
; CHECK-LABEL: @test8
; CHECK: mfspr [[REG1:[0-9]+]], 128
}

define i64 @test9() {
entry:
  %0 = tail call i64 @llvm.ppc.get.tfiar()
  ret i64 %0
; CHECK-LABEL: @test9
; CHECK: mfspr [[REG1:[0-9]+]], 129
}

declare void @llvm.ppc.set.texasr(i64)
declare void @llvm.ppc.set.texasru(i64)
declare void @llvm.ppc.set.tfhar(i64)
declare void @llvm.ppc.set.tfiar(i64)
declare i64 @llvm.ppc.get.texasr()
declare i64 @llvm.ppc.get.texasru()
declare i64 @llvm.ppc.get.tfhar()
declare i64 @llvm.ppc.get.tfiar()

define void @test10() {
entry:
  %0 = tail call i32 @llvm.ppc.tcheck()
  %1 = tail call i32 @llvm.ppc.treclaim(i32 5)
  %2 = tail call i32 @llvm.ppc.trechkpt()
  %3 = tail call i32 @llvm.ppc.tsr(i32 1)
  ret void
; CHECK-LABEL: @test10
; CHECK: tcheck [[REG1:[0-9]+]] 
; CHECK: treclaim. [[REG2:[0-9]+]] 
; CHECK: trechkpt. 
; CHECK: tresume.
}

declare i32 @llvm.ppc.tcheck()
declare i32 @llvm.ppc.treclaim(i32)
declare i32 @llvm.ppc.trechkpt()
declare i32 @llvm.ppc.tsr(i32)

