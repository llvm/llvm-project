; ============================================================================
; Reproduces the organic path where an available_externally
; comdat leader reaches the LTO optimization pipeline.
; If not handled properly, this would result in:
; "Declaration may not be in a Comdat! ptr @foo".
;
; Two modules share comdat $foo (leader @foo plus member @bar).
; During the LTO link @foo's non-prevailing copy is demoted to
; available_externally while the group is merged whole, so an
; available_externally leader survives in a comdat that still has a live member.
;
; ============================================================================

; The behavior is identical on COFF and ELF, so both triples share the same
; expectations (-mtriple keeps the triple out of the modules).

; REQUIRES: x86-registered-target

; RUN: split-file %s %t

; RUN: opt -mtriple=x86_64-pc-windows-msvc -thinlto-bc -o %t/w1.bc %t/prevailing.ll
; RUN: opt -mtriple=x86_64-pc-windows-msvc -thinlto-bc -o %t/w2.bc %t/importing.ll
; RUN: llvm-lto2 run -save-temps -o %t/w3 %t/w1.bc %t/w2.bc \
; RUN:   -r %t/w1.bc,foo,plx \
; RUN:   -r %t/w1.bc,bar,plx \
; RUN:   -r %t/w1.bc,keep,plx \
; RUN:   -r %t/w2.bc,foo,lx \
; RUN:   -r %t/w2.bc,bar,lx \
; RUN:   -r %t/w2.bc,use,plx
; RUN: llvm-dis -o - %t/w3.1.5.precodegen.bc | FileCheck %s

; RUN: opt -mtriple=x86_64-unknown-linux-gnu -thinlto-bc -o %t/l1.bc %t/prevailing.ll
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -thinlto-bc -o %t/l2.bc %t/importing.ll
; RUN: llvm-lto2 run -save-temps -o %t/l3 %t/l1.bc %t/l2.bc \
; RUN:   -r %t/l1.bc,foo,plx \
; RUN:   -r %t/l1.bc,bar,plx \
; RUN:   -r %t/l1.bc,keep,plx \
; RUN:   -r %t/l2.bc,foo,lx \
; RUN:   -r %t/l2.bc,bar,lx \
; RUN:   -r %t/l2.bc,use,plx
; RUN: llvm-dis -o - %t/l3.1.5.precodegen.bc | FileCheck %s

; The preserved comdat leader must be an emitted ODR definition (weak_odr) that
; still lives in its comdat.
; CHECK: define weak_odr {{.*}}@foo(){{.*}} comdat
; CHECK-NOT: available_externally{{.*}}@foo

;--- prevailing.ll
; Prevailing copies of the comdat group live here; @keep forces the group live.
$foo = comdat any

define linkonce_odr void @foo() comdat($foo) {
  call void @bar()
  ret void
}

define linkonce_odr void @bar() comdat($foo) {
  ret void
}

define void @keep() {
  call void @foo()
  ret void
}

;--- importing.ll
; Non-prevailing side. @foo is the leader (name == comdat). Its copy is demoted
; toward available_externally while the whole group is merged, so the leader can
; end up available_externally with a sibling member keeping the comdat
; non-discardable. @use keeps the group live.
$foo = comdat any

define linkonce_odr void @foo() comdat($foo) {
  call void @bar()
  ret void
}

define linkonce_odr void @bar() comdat($foo) {
  ret void
}

define void @use() {
  call void @foo()
  ret void
}
