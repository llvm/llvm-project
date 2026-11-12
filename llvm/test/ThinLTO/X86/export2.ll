; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/export2.ll -o %t2.bc
; 
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t3 \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t1.bc,callstaticfunc,px \
; RUN:  -r=%t2.bc,main,px \
; RUN:  -r=%t2.bc,callstaticfunc,

; both functions must appear as external linkage in the index.
; RUN: llvm-dis %t1.bc.thinlto.bc -o - | FileCheck %s
; CHECK: linkage: external
; CHECK: linkage: external

target triple = "x86_64-unknown-linux-gnu"

define void @callstaticfunc() {
entry:
  call void @staticfunc()
  ret void
}

define internal void @staticfunc() {
entry:
  ret void
}
