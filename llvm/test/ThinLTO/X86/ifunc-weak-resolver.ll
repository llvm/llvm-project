;; Simulate the frontend generating weak_odr definitions for an ifunc resolver
;; across multiple modules, with the linker resolving the definition in the 
;; first module as prevailing. Verify that ThinLink does not modify the linkage
;; of the ifunc resolver in the second module to available_externally.

; REQUIRES: x86-registered-target
; RUN: opt -module-summary %s -o %t1.o
; RUN: cp %t1.o %t2.o
; RUN: llvm-lto2 run %t1.o %t2.o -thinlto-emit-indexes -o %t.out \
; RUN:   -r=%t1.o,dispatch,px -r=%t1.o,resolver,px -r=%t1.o,ordinary,px \
; RUN:   -r=%t2.o,dispatch,x -r=%t2.o,resolver,x -r=%t2.o,ordinary,x
; RUN: llvm-dis %t2.o.thinlto.bc -o - | FileCheck %s

;; The second module is non-prevailing. Check that ThinLink drops its ordinary
;; definition to available_externally, but keeps the ifunc resolver's weak_odr
;; linkage.
; CHECK: guid: {{[0-9]+}}
; CHECK-SAME: linkage: available_externally
; CHECK: guid: {{[0-9]+}}
; CHECK-SAME: linkage: weak_odr

target triple = "x86_64-unknown-linux-gnu"
@dispatch = weak_odr ifunc void (), ptr @resolver

define weak_odr ptr @resolver() {
  ret ptr null
}

define weak_odr void @ordinary() {
  ret void
}
