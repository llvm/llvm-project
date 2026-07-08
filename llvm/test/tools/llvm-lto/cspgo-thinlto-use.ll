; REQUIRES: aarch64-registered-target
;
; RUN: llvm-profdata merge %S/Inputs/cspgo-cs.proftext -o %t.profdata
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-lto -thinlto-action=optimize \
; RUN:   -cs-profile-path=%t.profdata \
; RUN:   -o %t.opt.bc %t.o
; RUN: llvm-dis %t.opt.bc -o - | FileCheck %s
;
; CHECK: @main(){{.*}} !prof [[PROF:![0-9]+]]
; CHECK: CSProfileSummary
; CHECK: [[PROF]] = !{!"function_entry_count", i64 100000}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx14.0.0"

define i32 @main() {
entry:
  ret i32 0
}
