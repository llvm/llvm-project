; RUN: opt -S -passes=lower-target-intrinsics -mtriple=x86_64-unknown-linux -mcpu=x86-64 < %s | FileCheck %s

; Test that per-function target-cpu/target-features attributes are respected.
; The module default is x86-64, but individual functions override it.

; This function has haswell attributes — it should see avx2.
define i1 @func_with_haswell() #0 {
; CHECK-LABEL: @func_with_haswell(
; CHECK-NEXT:    ret i1 true
  %1 = call i1 @llvm.target.has.feature(metadata !"avx2")
  ret i1 %1
}

; This function has no overrides — uses module default (x86-64).
define i1 @func_with_default() {
; CHECK-LABEL: @func_with_default(
; CHECK-NEXT:    ret i1 false
  %1 = call i1 @llvm.target.has.feature(metadata !"avx2")
  ret i1 %1
}

; is.cpu respects per-function attributes too.
define i1 @func_is_haswell() #0 {
; CHECK-LABEL: @func_is_haswell(
; CHECK-NEXT:    ret i1 true
  %1 = call i1 @llvm.target.is.cpu(metadata !"haswell")
  ret i1 %1
}

define i1 @func_is_haswell_default() {
; CHECK-LABEL: @func_is_haswell_default(
; CHECK-NEXT:    ret i1 false
  %1 = call i1 @llvm.target.is.cpu(metadata !"haswell")
  ret i1 %1
}

attributes #0 = { "target-cpu"="haswell" }
