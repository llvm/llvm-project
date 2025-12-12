; RUN: opt < %s -passes=lower-allow-check -S | FileCheck %s
; RUN: opt < %s -passes=lower-allow-check -lower-allow-check-random-rate=0 -S | FileCheck %s

declare i1 @llvm.allow.sanitize.address()
declare i1 @llvm.allow.sanitize.thread()
declare i1 @llvm.allow.sanitize.memory()
declare i1 @llvm.allow.sanitize.hwaddress()

define i1 @test_address() sanitize_address {
; CHECK-LABEL: @test_address(
; CHECK-NEXT:    ret i1 true
  %1 = call i1 @llvm.allow.sanitize.address()
  ret i1 %1
}

define i1 @test_no_sanitize_address() {
; CHECK-LABEL: @test_no_sanitize_address(
; CHECK-NEXT:    ret i1 false
  %1 = call i1 @llvm.allow.sanitize.address()
  ret i1 %1
}

define i1 @test_address_but_no_thread() sanitize_address {
; CHECK-LABEL: @test_address_but_no_thread(
; CHECK-NEXT:    ret i1 false
  %1 = call i1 @llvm.allow.sanitize.thread()
  ret i1 %1
}

define i1 @test_thread() sanitize_thread {
; CHECK-LABEL: @test_thread(
; CHECK-NEXT:    ret i1 true
  %1 = call i1 @llvm.allow.sanitize.thread()
  ret i1 %1
}

define i1 @test_no_sanitize_thread() {
; CHECK-LABEL: @test_no_sanitize_thread(
; CHECK-NEXT:    ret i1 false
  %1 = call i1 @llvm.allow.sanitize.thread()
  ret i1 %1
}

define i1 @test_memory() sanitize_memory {
; CHECK-LABEL: @test_memory(
; CHECK-NEXT:    ret i1 true
  %1 = call i1 @llvm.allow.sanitize.memory()
  ret i1 %1
}

define i1 @test_no_sanitize_memory() {
; CHECK-LABEL: @test_no_sanitize_memory(
; CHECK-NEXT:    ret i1 false
  %1 = call i1 @llvm.allow.sanitize.memory()
  ret i1 %1
}

define i1 @test_hwaddress() sanitize_hwaddress {
; CHECK-LABEL: @test_hwaddress(
; CHECK-NEXT:    ret i1 true
  %1 = call i1 @llvm.allow.sanitize.hwaddress()
  ret i1 %1
}

define i1 @test_no_sanitize_hwaddress() {
; CHECK-LABEL: @test_no_sanitize_hwaddress(
; CHECK-NEXT:    ret i1 false
  %1 = call i1 @llvm.allow.sanitize.hwaddress()
  ret i1 %1
}
