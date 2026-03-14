; RUN: llvm-as %S/Inputs/f.ll -o %t.f.bc
; RUN: llvm-as %S/Inputs/g.ll -o %t.g.bc
; RUN: llvm-as %S/Inputs/i.ll -o %t.i.bc
; RUN: rm -f %t.lib
; RUN: llvm-ar cr %t.lib %t.f.bc %t.g.bc %t.i.bc
; RUN: llvm-link %s %t.lib -o %t.linked.bc --only-needed
; RUN: llvm-nm %t.linked.bc | FileCheck %s

; CHECK: -------- T f
; CHECK: -------- T i

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@i = external global ptr
@llvm.used = appending global [1 x ptr] [ptr @i], section "llvm.metadata"
