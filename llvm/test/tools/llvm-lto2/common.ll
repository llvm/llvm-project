; REQUIRES: X86
; RUN: llvm-as < %s > %t1.bc
; RUN: llvm-as < %p/Inputs/common.ll > %t2.bc

; Test that the common merging (size + alignment) is properly handled

; Client marked the "large with little alignment" one as prevailing
; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,px \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck %s

; Same as before, but reversing the order of the inputs
; RUN: llvm-lto2 %t2.bc %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,px \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck %s


; Client marked the "small with large alignment" one as prevailing
; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,px \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck %s

; Same as before, but reversing the order of the inputs
; RUN: llvm-lto2 %t2.bc %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,px \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck  %s


; Client didn't mark any as prevailing, we keep the first one we see as "external"
; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck  %s

; Same as before, but reversing the order of the inputs
; RUN: llvm-lto2 %t2.bc %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,v,x \
; RUN:  -r %t2.bc,v,x \
; RUN:  -r %t1.bc,foo,px \
; RUN:  -r %t2.bc,bar,px
; RUN: llvm-dis < %t.o.0.0.preopt.bc | FileCheck  %s

target triple = "x86_64-apple-macosx10.11.0"

@v = common global i8 0, align 8


; CHECK: @v = common global [2 x i8] zeroinitializer, align 8

define i8 *@foo() {
 ret i8 *@v
}