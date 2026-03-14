; Test to ensure that non-prevailing weak aliasee is kept as a weak definition
; when the alias is not dead.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: llvm-lto2 run %t1.bc \
; RUN:	 -r=%t1.bc,__a,lx \
; RUN:	 -r=%t1.bc,__b,l \
; RUN:	 -r=%t1.bc,a,plx \
; RUN:	 -r=%t1.bc,b,pl \
; RUN:   -o %t2.o -save-temps

; Check that __a is kept as a weak def. __b can be dropped since its alias is
; not live and will also be dropped.
; RUN: llvm-dis %t2.o.1.1.promote.bc -o - | FileCheck %s
; CHECK: define weak hidden void @__a
; CHECK: declare hidden void @__b
; CHECK: declare void @b

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = hidden alias void (), ptr @__a

define weak hidden void @__a() {
entry:
  ret void
}

@b = hidden alias void (), ptr @__b

define weak hidden void @__b() {
entry:
  ret void
}
