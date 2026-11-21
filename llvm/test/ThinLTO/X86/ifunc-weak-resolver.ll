;; Simulate the frontend generating weak_odr definitions for an ifunc resolver
;; across multiple modules, with the linker resolving the definition in the 
;; first module as prevailing. Verify that the backend retains its original
;; weak_odr linkage during the import phase to prevent verifier failures.

; REQUIRES: x86-registered-target
; RUN: opt -module-summary %s -o %t1.o
; RUN: cp %t1.o %t2.o
; RUN: llvm-lto2 run %t1.o %t2.o -thinlto-emit-indexes -save-temps -o %t.out \
; RUN:   -r=%t1.o,dispatch,px -r=%t1.o,resolver,px -r=%t1.o,ordinary,px \
; RUN:   -r=%t2.o,dispatch,x -r=%t2.o,resolver,x -r=%t2.o,ordinary,x
; RUN: llvm-dis %t2.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX
; RUN: llvm-dis %t.out.2.1.promote.bc -o - | FileCheck %s --check-prefix=IR

;; Thin link marks both non-prevailing definitions available_externally.
; INDEX-COUNT-2: linkage: available_externally
;; Backend keeps the ifunc resolver's weak_odr definition and comdat.
; IR: define weak_odr{{.*}} ptr @resolver(){{.*}} comdat
; IR: define available_externally{{.*}} void @ordinary()

target triple = "x86_64-unknown-linux-gnu"
$resolver = comdat any
@dispatch = weak_odr ifunc void (), ptr @resolver

define weak_odr ptr @resolver() comdat {
  ret ptr null
}

define weak_odr void @ordinary() {
  ret void
}
