; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/funcimport2.ll -o %t2.bc

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,_foo,plx \
; RUN:     -r=%t2.bc,_main,plx \
; RUN:     -r=%t2.bc,_foo,l
; RUN: llvm-dis %t.o.1.3.import.bc -o - | FileCheck %s
; CHECK: define available_externally void @foo()

; We shouldn't do any importing at -O0
; rm -f %t.o.1.3.import.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -O0 \
; RUN:     -r=%t1.bc,_foo,plx \
; RUN:     -r=%t2.bc,_main,plx \
; RUN:     -r=%t2.bc,_foo,l
; RUN: llvm-dis %t.o.1.3.import.bc -o - | FileCheck %s --check-prefix=CHECKO0
; CHECKO0: declare void @foo(...)

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @foo() #0 {
entry:
  ret void
}
