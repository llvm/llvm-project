; Function import can promote an internal function to external but not mark it as prevailing.
; Given that the internal function's attributes would have already propagated to its callers 
; that are part of the import chain there's no need to actually propagate off this copy as 
; propagating the caller performs the same thing.
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/callees.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 \
; RUN:   %t1.bc %t2.bc -o %t.o \
; RUN:   -r %t1.bc,caller,l -r %t1.bc,caller_noattr,l -r %t1.bc,importer,px -r %t1.bc,importer_noattr,px \
; RUN:   -r %t2.bc,caller,px -r %t2.bc,caller_noattr,px \
; RUN:   -save-temps
; RUN: llvm-dis -o - %t.o.1.3.import.bc | FileCheck %s --match-full-lines

;--- main.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @caller()
declare void @caller_noattr()

; CHECK: define void @importer() [[ATTR_PROP:#[0-9]+]] {
define void @importer() {
  call void @caller()
  ret void
}

; If somehow the caller doesn't get the attributes, we
; shouldn't propagate from the internal callee.
; CHECK: define void @importer_noattr() {
define void @importer_noattr() {
  call void @caller_noattr()
  ret void
}

; CHECK: define available_externally hidden void @callee{{.*}}

; CHECK-DAG: attributes [[ATTR_PROP]] = { norecurse nounwind }

;--- callees.ll
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

attributes #0 = { nounwind norecurse }

define void @caller() #0 {
  call void @callee()
  ret void
}

define void @caller_noattr() {
  call void @callee()
  ret void
}

define internal void @callee() #0 {
  ret void
}
