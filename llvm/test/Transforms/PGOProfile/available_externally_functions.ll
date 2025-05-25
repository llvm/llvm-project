; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN --check-prefix=GEN-COMDAT

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN-COMDAT: $__llvm_profile_raw_version = comdat any
; GEN-COMDAT: @__llvm_profile_raw_version = hidden constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_foo = linkonce_odr hidden constant [3 x i8] c"foo"
; GEN: @__profn_bar = linkonce_odr hidden constant [3 x i8] c"bar"

define available_externally hidden void @foo() {
  ret void
}

define available_externally i32 @bar() {
  ret i32 42
}
