; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/linkonce_aliasee_ref_import.ll -o %t2.bc

; Import with instr limit to ensure only foo imported.
; RUN: llvm-lto -thinlto-action=run -exported-symbol=main -import-instr-limit=5 %t1.bc %t2.bc
; RUN: llvm-nm -o - < %t1.bc.thinlto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm -o - < %t2.bc.thinlto.o | FileCheck %s --check-prefix=NM2

; Import with instr limit to ensure only foo imported.
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:    -r=%t1.bc,foo,pxl \
; RUN:    -r=%t1.bc,baz,pxl \
; RUN:    -r=%t1.bc,baz.clone,pxl \
; RUN:    -r=%t1.bc,bar,pl \
; RUN:    -r=%t2.bc,main,pxl \
; RUN:    -r=%t2.bc,foo,l \
; RUN:    -import-instr-limit=5
; RUN: llvm-nm -o - < %t1.bc.thinlto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm -o - < %t2.bc.thinlto.o | FileCheck %s --check-prefix=NM2

; Check that we converted baz.clone to a weak
; NM1: W baz.clone

; Check that we imported a ref (and not def) to baz.clone
; NM2: U baz.clone

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

$baz.clone = comdat any
@baz = weak alias void (), ptr @baz.clone

define void @foo() #5 align 2 {
  tail call void @baz.clone()
  ret void
}
define linkonce_odr void @baz.clone() #5 comdat align 2 {
  call void @bar()
  call void @bar()
  call void @bar()
  call void @bar()
  call void @bar()
  call void @bar()
  call void @bar()
  ret void
}

define void @bar() {
  ret void
}
