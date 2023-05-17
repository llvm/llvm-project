; RUN: opt < %s -passes='thinlto-pre-link<O2>' --cs-profilegen-file=file -cspgo-kind=cspgo-instr-gen-pipeline -S | FileCheck %s

; CHECK: $__llvm_profile_filename = comdat any
; CHECK: @__llvm_profile_filename = hidden local_unnamed_addr constant [5 x i8] c"file\00", comdat

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test() {
  ret i32 0
}
