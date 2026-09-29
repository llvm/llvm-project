; RUN: llvm-as %s -o=%t1
; RUN: rm -f %t2
; RUN: llvm-ar rcs %t2 %t1
; RUN: llvm-nm --print-armap %t2 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global global_asm_sym"
module asm "global_asm_sym:"
module asm "local_asm_sym:"
module asm ".long undef_asm_sym"

; CHECK: Archive map
; CHECK-NEXT: global_asm_sym in archive-ir-asm.ll

; CHECK: archive-ir-asm.ll
; CHECK-NEXT:         T global_asm_sym
; CHECK-NEXT:         t local_asm_sym
; CHECK-NEXT:         U undef_asm_sym
