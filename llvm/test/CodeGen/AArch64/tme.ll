; RUN: llc %s -verify-machineinstrs -o - | FileCheck %s

target triple = "aarch64"

define i64 @test_tstart() #0 {
  %r = tail call i64 @llvm.aarch64.tstart()
  ret i64 %r
}
declare i64 @llvm.aarch64.tstart() #1
; CHECK-LABEL: test_tstart
; CHECK: tstart x

define i64 @test_ttest() #0 {
  %r = tail call i64 @llvm.aarch64.ttest()
  ret i64 %r
}
declare i64 @llvm.aarch64.ttest() #1
; CHECK-LABEL: test_ttest
; CHECK: ttest x

define void @test_tcommit() #0 {
  tail call void @llvm.aarch64.tcommit()
  ret void
}
declare void @llvm.aarch64.tcommit() #1
; CHECK-LABEL: test_tcommit
; CHECK: tcommit

define void @test_tcancel() #0 {
  tail call void @llvm.aarch64.tcancel(i64 0) #1
  tail call void @llvm.aarch64.tcancel(i64 1) #1
  tail call void @llvm.aarch64.tcancel(i64 65534) #1
  tail call void @llvm.aarch64.tcancel(i64 65535) #1
  ret void
}
declare void @llvm.aarch64.tcancel(i64 immarg) #1
; CHECK-LABEL: test_tcancel
; CHECK: tcancel #0
; CHECK: tcancel #0x1
; CHECK: tcancel #0xfffe
; CHECK: tcancel #0xffff

attributes #0 = { "target-features"="+tme" }
attributes #1 = { nounwind }
