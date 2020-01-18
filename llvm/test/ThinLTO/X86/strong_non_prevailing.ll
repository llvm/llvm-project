; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/strong_non_prevailing.ll -o %t2.bc

; RUN: llvm-lto -thinlto-action=run %t.bc %t2.bc -exported-symbol=__llvm_profile_filename
; RUN: llvm-nm -o - < %t.bc.thinlto.o | FileCheck %s --check-prefix=EXPORTED
; RUN: llvm-nm -o - < %t2.bc.thinlto.o 2>&1 | FileCheck %s --check-prefix=NOT_EXPORTED

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__llvm_profile_filename = comdat any

@__llvm_profile_filename = constant [19 x i8] c"default_%m.profraw\00", comdat

; EXPORTED: R __llvm_profile_filename
; NOT_EXPORTED-NOT: R __llvm_profile_filename
