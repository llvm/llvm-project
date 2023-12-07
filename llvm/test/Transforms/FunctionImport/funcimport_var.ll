;; This test makes sure a static var is not selected as a callee target (it will
;; have a different GUID). Similarly, ensure that a callee GUID annotated from a
;; sample pgo profile that matches a global variable GUID (either due to code
;; changes or a hash collision), does not get considered for importing. Either
;; of these can crash compilation.
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_var2.ll -o %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=thinlink -o %t3 %t.bc %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=import -thinlto-index=%t3 %t.bc %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=run %t.bc %t2.bc -exported-symbol=_Z4LinkPKcS0_
; RUN: llvm-nm %t.bc.thinlto.o | FileCheck %s --implicit-check-not=globalvar
; RUN: llvm-lto2 run %t.bc %t2.bc -o %t.out \
; RUN:   -r %t.bc,_Z4LinkPKcS0_,plx \
; RUN:   -r %t.bc,link,l \
; RUN:   -r %t2.bc,globalvar,plx \
; RUN:   -r %t2.bc,get_link,plx
; RUN: llvm-nm %t.out.1 | FileCheck %s --implicit-check-not=globalvar
; CHECK: U link

; REQUIRES: x86-registered-target

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z4LinkPKcS0_(ptr, ptr) local_unnamed_addr !prof !1 {
  %3 = tail call i32 @link(ptr %0, ptr %1) #2
  ret i32 %3
}

; Function Attrs: nounwind
declare i32 @link(ptr, ptr) local_unnamed_addr

;; This matches the GUID of global variable @globalvar in the other input.
!1 = !{!"function_entry_count", i64 110, i64 12887606300320728018}
