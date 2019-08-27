; RUN: llvm-as < %s > %t1.bc
; RUN: llvm-as < %p/Inputs/common2.ll > %t2.bc

; Test that the common merging (size + alignment) is properly handled

; Client marked the "large with little alignment" one as prevailing
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,px \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck %s --check-prefix=LARGE-PREVAILED

; Same as before, but reversing the order of the inputs
; RUN: llvm-lto2 run %t2.bc %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,px \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck %s --check-prefix=LARGE-PREVAILED

; Client marked the "small with large alignment" one as prevailing
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,px \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck %s --check-prefix=SMALL-PREVAILED

; Same as before, but reversing the order of the inputs
; RUN: llvm-lto2 run %t2.bc %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,px \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck  %s --check-prefix=SMALL-PREVAILED


; Client didn't mark any as prevailing, we keep the first one we see as "external"
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck  %s --check-prefix=NONE-PREVAILED1

; Same as before, but reversing the order of the inputs
; RUN: llvm-lto2 run %t2.bc %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck  %s --check-prefix=NONE-PREVAILED2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@v = common global i8 0, align 8

; LARGE-PREVAILED: @v = common global i16 0, align 8
; SMALL-PREVAILED: @v = common global [2 x i8] zeroinitializer, align 8
; BOTH-PREVAILED1: @v = common global i16 0, align 8
; BOTH-PREVAILED2: common global [2 x i8] zeroinitializer, align 8
; In this case the first is kept as external
; NONE-PREVAILED1: @v = external global i8, align 8
; NONE-PREVAILED2: @v = external global i16, align 4

define i8 *@foo() {
 ret i8 *@v
}
