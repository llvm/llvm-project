; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/funcimport_alwaysinline.ll -o %t2.bc

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,foo,plx \
; RUN:     -r=%t2.bc,main,plx \
; RUN:     -r=%t2.bc,foo,l \
; RUN:     -import-instr-limit=0
; RUN: llvm-dis %t.o.2.3.import.bc -o - | FileCheck %s

; foo() being always_inline should be imported irrespective of the
; instruction limit
; CHECK: define available_externally dso_local void @foo()

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: alwaysinline nounwind uwtable
define void @foo() #0 {
entry:
  ret void
}

attributes #0 = { alwaysinline nounwind uwtable }
