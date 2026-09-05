; REQUIRES: x86-registered-target

;; The prevailing raw-version definition may have been emitted before LTO by a
;; frontend using an older raw format. Backend-only CS temporal instrumentation
;; must refresh that initializer to describe the records it actually emits.
;; The non-prevailing copy must remain a declaration.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %S/Inputs/thinlto_cspgo_raw_version_nonprevailing.ll -o %t2.bc
; RUN: llvm-lto2 run -lto-cspgo-profile-file=alloc -lto-cspgo-gen \
; RUN:   -pgo-temporal-instrumentation -save-temps -o %t %t1.bc %t2.bc \
; RUN:   -r=%t1.bc,main,plx \
; RUN:   -r=%t1.bc,__llvm_profile_raw_version,plx \
; RUN:   -r=%t2.bc,__llvm_profile_raw_version,x
; RUN: llvm-dis %t.1.4.opt.bc -o - | FileCheck %s --check-prefix=PREVAILING
; RUN: llvm-dis %t.2.4.opt.bc -o - | FileCheck %s --check-prefix=NONPREVAILING

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__llvm_profile_raw_version = comdat any

;; Raw version 10 with the IR and CSIR variant masks. The backend uses raw
;; version 11 and adds the temporal-profile variant mask.
@__llvm_profile_raw_version = hidden constant i64 216172782113783818, comdat
@llvm.compiler.used = appending global [1 x ptr] [ptr @__llvm_profile_raw_version], section "llvm.metadata"

; PREVAILING: @__llvm_profile_raw_version = hidden constant i64 -9007199254740991989
; NONPREVAILING: @__llvm_profile_raw_version = external hidden constant i64

define i32 @main() {
entry:
  ret i32 0
}
