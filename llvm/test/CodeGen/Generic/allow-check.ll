; RUN: llc < %s -O3
; RUN: llc < %s -O3 -global-isel
; RUN: llc < %s -O3 -fast-isel

; RUN: llc < %s -O0
; RUN: llc < %s -O0 -global-isel
; RUN: llc < %s -O0 -fast-isel

define i1 @test_runtime() local_unnamed_addr {
entry:
  %allow = call i1 @llvm.allow.runtime.check(metadata !"test_check")
  ret i1 %allow
}

declare i1 @llvm.allow.runtime.check(metadata) nounwind

define i1 @test_ubsan() local_unnamed_addr {
entry:
  %allow = call i1 @llvm.allow.ubsan.check(i8 7)
  ret i1 %allow
}

declare i1 @llvm.allow.ubsan.check(i8) nounwind
