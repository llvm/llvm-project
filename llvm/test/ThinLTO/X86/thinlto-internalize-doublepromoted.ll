; Test to ensure that we can internalize values produced from two rounds
; of ThinLTO promotion, so they end up with two ".llvm.${hash}" suffixes.
; Only the second should be stripped when consulting the index to locate the
; summary.
;
; Note that this cannot happen currently via clang, but in other use cases such
; as the Rust compiler which does a first round of ThinLTO on library code,
; producing bitcode, and a second round on the final binary.
;
; In this case we assume a prior round of ThinLTO has promoted @foo, and
; subsequent optimization created an internal switch table expansion variable
; that is internal and contains the promoted name of the enclosing function.
; This variable will be promoted in the second round of ThinLTO if @foo is
; imported again.

; RUN: opt -module-summary -o %t.bc %s
; RUN: opt -module-summary -o %t-main.bc %S/Inputs/thinlto-internalize-doublepromoted.ll
; RUN: llvm-lto -thinlto-action=thinlink %t.bc %t-main.bc -o %t-index.bc
; RUN: llvm-lto -thinlto-action=internalize -exported-symbol=main -thinlto-index %t-index.bc %t.bc -o %t.internalize.bc
; RUN: llvm-dis %t.internalize.bc -o - | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

; CHECK: @switch.table.foo.llvm.123.llvm.0 = hidden unnamed_addr constant
@switch.table.foo.llvm.123 = private unnamed_addr constant [10 x i8] c"\00\01\02\03\00\00\00\00\00\09", align 1

; CHECK: define hidden void @foo.llvm.123()
define hidden void @foo.llvm.123() {
  %1 = getelementptr inbounds [10 x i8], [10 x i8]* @switch.table.foo.llvm.123, i64 0, i64 0
  store i8 1, i8* %1, align 8
  ret void
}
