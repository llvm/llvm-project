;; Simulate the frontend generating weak_odr definitions for an ifunc resolver
;; across multiple modules, with the linker resolving the definition in the
;; first module as prevailing. Verify that the backend retains its original
;; weak_odr linkage during ThinLTO finalization to prevent verifier failures.

; REQUIRES: x86-registered-target
; RUN: opt -module-summary %s -o %t1.o
; RUN: cp %t1.o %t2.o
; RUN: llvm-lto2 run %t1.o %t2.o -thinlto-emit-indexes -save-temps -o %t.out \
; RUN:   -r=%t1.o,dispatch,px -r=%t1.o,shared_dispatch,px -r=%t1.o,resolver,px -r=%t1.o,ordinary,px \
; RUN:   -r=%t1.o,weak_dispatch,px -r=%t1.o,weak_resolver,px \
; RUN:   -r=%t2.o,dispatch,x -r=%t2.o,shared_dispatch,x -r=%t2.o,resolver,x -r=%t2.o,ordinary,x \
; RUN:   -r=%t2.o,weak_dispatch,x -r=%t2.o,weak_resolver,x
; RUN: llvm-dis %t2.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX
; RUN: llvm-dis %t.out.2.1.promote.bc -o - | FileCheck %s --check-prefix=IR --implicit-check-not='comdat($ordinary)'

;; Thin link marks the non-prevailing weak_odr and weak definitions available_externally.
; INDEX-COUNT-3: linkage: available_externally
; IR: @weak_dispatch = weak ifunc void (), ptr @weak_resolver
;; Backend keeps the ifunc resolver's weak_odr definition and comdat.
; IR: define weak_odr{{.*}} ptr @resolver(){{.*}} comdat
;; Keep the interposable weak resolver's definition instead of converting it
;; to a declaration.
; IR: define weak {{.*}}ptr @weak_resolver()

;; A local resolver sharing a non-prevailing leader's COMDAT also stays a
;; definition.
; IR: define available_externally{{.*}} void @ordinary()
; IR: define internal{{.*}} ptr @shared_resolver()

target triple = "x86_64-unknown-linux-gnu"
$resolver = comdat any
$ordinary = comdat any
@dispatch = weak_odr ifunc void (), ptr @resolver
@shared_dispatch = weak_odr ifunc void (), ptr @shared_resolver
@weak_dispatch = weak ifunc void (), ptr @weak_resolver

define weak_odr ptr @resolver() comdat {
  ret ptr null
}

define weak ptr @weak_resolver() {
  ret ptr null
}

define weak_odr void @ordinary() comdat {
  ret void
}

define internal ptr @shared_resolver() comdat($ordinary) {
  ret ptr null
}
