; REQUIRES: x86-registered-target

; RUN: opt %s -S | FileCheck --check-prefix=CHECK-FILE %s
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: declare

; CHECK-FILE: ModuleID
; CHECK-FILE: source_filename
; CHECK-FILE: datalayout
; CHECK-FILE: triple
; CHECK-FILE: module asm
; CHECK-FILE: declare void @g

; CHECK-FINAL-NOT: ModuleID
; CHECK-FINAL-NOT: source_filename
; CHECK-FINAL-NOT: datalayout
; CHECK-FINAL-NOT: triple
; CHECK-FINAL-NOT: module asm
; CHECK-FINAL: declare void @g

source_filename = "/tmp/a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
module asm "foo"

declare void @g()
