; Callee1 isn't defined, propagation goes conservative
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/callees.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 %t1.bc %t2.bc -o %t.o -r %t1.bc,caller,px -r %t1.bc,callee,l -r %t1.bc,callee1,l -r %t2.bc,callee,px -save-temps
; RUN: llvm-dis -o - %t.o.1.3.import.bc | FileCheck %s

;--- main.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @callee()
declare void @callee1()

; CHECK-NOT: Function Attrs: 
; CHECK: define void @caller()
define void @caller() {
  call void @callee()
  call void @callee1()
  ret void
}

;--- callees.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

attributes #0 = { nounwind norecurse }

define void @callee() #0 {
  ret void
}
