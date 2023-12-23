; RUN: split-file %s %t
; RUN: opt %t/asm_global.ll -o %t/asm_global.bc
; RUN: opt %t/asm_weak.ll -o %t/asm_weak.bc
; RUN: opt %t/fn_global.ll -o %t/fn_global.bc
; RUN: opt %t/fn_weak.ll -o %t/fn_weak.bc

; RUN: not llvm-lto %t/asm_global.bc %t/fn_global.bc -o %to1.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llvm-lto %t/fn_global.bc %t/asm_global.bc -o %to1.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR

; RUN: llvm-lto %t/asm_global.bc %t/fn_weak.bc -o %to2.o --save-linked-module --exported-symbol=foo
; RUN: llvm-dis < %to2.o.linked.bc | FileCheck %s --check-prefix=ASM_GLOBAL
; RUN: llvm-lto %t/fn_weak.bc %t/asm_global.bc -o %to2.o --save-linked-module --exported-symbol=foo
; RUN: llvm-dis < %to2.o.linked.bc | FileCheck %s --check-prefix=ASM_GLOBAL

; FIXME: We want to drop the weak asm symbol.
; RUN: not llvm-lto %t/asm_global.bc %t/asm_weak.bc -o %to3.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llvm-lto %t/asm_weak.bc %t/asm_global.bc -o %to3.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR

; FIXME: We want to drop the weak asm symbol.
; RUN: not llvm-lto %t/asm_weak.bc %t/fn_global.bc -o %to4.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR2
; RUN: not llvm-lto %t/fn_global.bc %t/asm_weak.bc -o %to4.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR2

; RUN: not llvm-lto %t/asm_weak.bc %t/fn_weak.bc -o %to5.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llvm-lto %t/fn_weak.bc %t/asm_weak.bc -o %to5.o --save-linked-module --exported-symbol=foo 2>&1 | FileCheck %s --check-prefix=ERR

; Drop the weak function.
; RUN: llvm-lto %t/fn_global.bc %t/fn_weak.bc -o %to6.o --save-linked-module --exported-symbol=foo
; RUN: llvm-dis < %to6.o.linked.bc | FileCheck %s --check-prefix=FN_GLOBAL
; RUN: llvm-lto %t/fn_weak.bc %t/fn_global.bc -o %to6.o --save-linked-module --exported-symbol=foo
; RUN: llvm-dis < %to6.o.linked.bc | FileCheck %s --check-prefix=FN_GLOBAL

; ERR: error: <unknown>:0: symbol 'foo' is already defined
; ERR2: error: <unknown>:0: foo changed binding to STB_GLOBAL

; FN_GLOBAL: define i32 @foo(i32 %i)

; ASM_GLOBAL: module asm ".global foo"
; ASM_GLOBAL-NOT: define i32 @foo(i32 %i)

;--- asm_global.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global foo"
module asm "foo:"
module asm "leal	1(%rdi), %eax"
module asm "retq"

;--- asm_weak.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".weak foo"
module asm "foo:"
module asm "leal	2(%rdi), %eax"
module asm "retq"

;--- fn_global.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %i) {
  %r = add i32 %i, 3
  ret i32 %r
}

define internal i32 @bar(i32 %i) {
  %r = call i32 @foo(i32 %i)
  ret i32 %r
}

;--- fn_weak.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define weak i32 @foo(i32 %i) {
  %r = add i32 %i, 4
  ret i32 %r
}

define internal i32 @bar(i32 %i) {
  %r = call i32 @foo(i32 %i)
  ret i32 %r
}
