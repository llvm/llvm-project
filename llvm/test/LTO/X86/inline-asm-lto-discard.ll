; Check that non-prevailing symbols in module inline assembly are discarded
; during regular LTO otherwise the final symbol binding could be wrong.

; RUN: split-file %s %t
; RUN: opt %t/t1.ll -o %t1
; RUN: opt %t/t2.ll -o %t2
; RUN: opt %t/t3.ll -o %t3
; RUN: opt %t/t4.ll -o %t4

; RUN: llvm-lto2 run -o %to1 -save-temps %t1 %t2 \
; RUN:  -r %t1,foo,px \
; RUN:  -r %t2,foo, \
; RUN:  -r %t2,bar,pl
; RUN: llvm-dis < %to1.0.0.preopt.bc | FileCheck %s --check-prefix=ASM1
; RUN: llvm-nm %to1.0 | FileCheck %s --check-prefix=SYM
; RUN: llvm-objdump -d --disassemble-symbols=foo %to1.0 \
; RUN:   | FileCheck %s --check-prefix=DEF

; RUN: llvm-lto2 run -o %to2 -save-temps %t2 %t3 \
; RUN:  -r %t2,foo, \
; RUN:  -r %t2,bar,pl \
; RUN:  -r %t3,foo,px
; RUN: llvm-dis < %to2.0.0.preopt.bc | FileCheck %s --check-prefix=ASM2
; RUN: llvm-nm %to2.0 | FileCheck %s --check-prefix=SYM
; RUN: llvm-objdump -d --disassemble-symbols=foo %to2.0 \
; RUN:   | FileCheck %s --check-prefix=DEF

; Check that ".symver" is properly handled.
; RUN: llvm-lto2 run -o %to3 -save-temps %t4 \
; RUN:  -r %t4,bar, \
; RUN:  -r %t4,foo, \
; RUN:  -r %t4,foo@@VER1,px
; RUN: llvm-dis < %to3.0.0.preopt.bc | FileCheck %s --check-prefix=ASM3

; ASM1:      module asm ".lto_discard foo"
; ASM1-NEXT: module asm ".weak foo"
; ASM1-NEXT: module asm ".equ foo,bar"

; ASM2:      module asm ".lto_discard foo"
; ASM2-NEXT: module asm ".weak foo"
; ASM2-NEXT: module asm ".equ foo,bar"
; ASM2-NEXT: module asm ".lto_discard"
; ASM2-NEXT: module asm " .global foo ; foo: leal    2(%rdi), %eax"

; ASM3-NOT:  module asm ".lto_discard foo"

; SYM: T foo

; DEF: leal    2(%rdi), %eax

;--- t1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @foo(i32 %0) {
  %2 = add nsw i32 %0, 2
  ret i32 %2
}

;--- t2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".weak foo"
module asm ".equ foo,bar"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (i32 (i32)* @bar to i8*)], section "llvm.metadata"

define internal i32 @bar(i32 %0) {
  %2 = add nsw i32 %0, 1
  ret i32 %2
}

;--- t3.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm " .global foo ; foo: leal    2(%rdi), %eax"

;--- t4.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global foo"
module asm "foo: call bar"
module asm ".symver foo,foo@@@VER1"
module asm ".symver bar,bar@@@VER1"
