; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/module_asm.ll -o %t2.bc

; RUN: llvm-lto -thinlto-action=run -exported-symbol=main %t1.bc %t2.bc
; RUN: llvm-nm %t1.bc.thinlto.o | FileCheck  %s --check-prefix=NM0
; RUN: llvm-nm %t2.bc.thinlto.o | FileCheck  %s --check-prefix=NM1

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,foo,lx \
; RUN:     -r=%t1.bc,foo,plx \
; RUN:     -r=%t1.bc,_simplefunction,pl \
; RUN:     -r=%t2.bc,main,plx \
; RUN:     -r=%t2.bc,_simplefunction,l
; RUN: llvm-nm %t.o.1 | FileCheck  %s --check-prefix=NM0
; RUN: llvm-nm %t.o.2 | FileCheck  %s --check-prefix=NM1

; NM0: T foo
; NM1-NOT: foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.text"
module asm "\09.globl\09foo"
module asm "\09.align\0916, 0x90"
module asm "\09.type\09foo,@function"
module asm "foo:"
module asm "\09ret "
module asm "\09.size\09foo, .-foo"
module asm ""

declare zeroext i16 @foo() #0

define i32 @_simplefunction() #1 {
  ret i32 1
}
