; REQUIRES: aarch64-registered-target
;
; RUN: llvm-profdata merge %S/Inputs/cspgo-noncs.proftext -o %t.profdata
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-lto -thinlto-action=optimize \
; RUN:   -cs-profile-generate -cs-profile-path=%t.profdata \
; RUN:   -o %t.opt.bc %t.o
; RUN: llvm-dis %t.opt.bc -o - | FileCheck %s
;
; CHECK: @__profc_main = private global
; CHECK-LABEL: @main()
; CHECK:   %pgocount = load i64, ptr @__profc_main
; CHECK:   store i64 %{{.*}}, ptr @__profc_main

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx14.0.0"

define i32 @main() {
entry:
  ret i32 0
}
