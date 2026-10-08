; REQUIRES: aarch64-registered-target
;
; RUN: llvm-profdata merge %S/Inputs/cspgo-noncs.proftext -o %t.profdata
; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto -cs-profile-generate -cs-profile-path=%t.profdata \
; RUN:   -exported-symbol=_main -o %t.lto.o %t.o
; RUN: llvm-nm %t.lto.o | FileCheck %s
;
; CHECK: __profc_main

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx14.0.0"

define i32 @main() {
entry:
  ret i32 0
}
