; RUN: llc -mtriple=avr -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: test_nop:
; CHECK: nop
; CHECK: ret
define void @test_nop() {
  call void @llvm.avr.nop()
  ret void
}

; CHECK-LABEL: test_sei:
; CHECK: sei
; CHECK: ret
define void @test_sei() {
  call void @llvm.avr.sei()
  ret void
}

; CHECK-LABEL: test_cli:
; CHECK: cli
; CHECK: ret
define void @test_cli() {
  call void @llvm.avr.cli()
  ret void
}

; CHECK-LABEL: test_sleep:
; CHECK: sleep
; CHECK: ret
define void @test_sleep() {
  call void @llvm.avr.sleep()
  ret void
}

; CHECK-LABEL: test_wdr:
; CHECK: wdr
; CHECK: ret
define void @test_wdr() {
  call void @llvm.avr.wdr()
  ret void
}

; CHECK-LABEL: test_swap:
; CHECK: swap r24
; CHECK: ret
define i8 @test_swap(i8 %a) {
  %1 = call i8 @llvm.avr.swap(i8 %a)
  ret i8 %1
}

declare void @llvm.avr.nop()
declare void @llvm.avr.sei()
declare void @llvm.avr.cli()
declare void @llvm.avr.sleep()
declare void @llvm.avr.wdr()
declare i8 @llvm.avr.swap(i8)
