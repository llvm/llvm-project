;; The test in this file do not appear in tls-models.ll because
;; they are not auto-generated.
; RUN: llc -mtriple=riscv64 -relocation-model=pic -enable-tlsdesc < %s \
; RUN:     | llvm-mc -triple=riscv64 -filetype=obj -o - \
; RUN:     | llvm-readelf --symbols - \
; RUN:     | FileCheck %s

; RUN: llc -mtriple=riscv32 -relocation-model=pic -enable-tlsdesc < %s \
; RUN:     | llvm-mc -triple=riscv32 -filetype=obj -o - \
; RUN:     | llvm-readelf --symbols - \
; RUN:     | FileCheck %s

; Check that TLS symbols are lowered correctly based on the specified
; model. Make sure they're external to avoid them all being optimised to Local
; Exec for the executable.

@unspecified = external thread_local global i32

define ptr @f1() nounwind {
entry:
  ret ptr @unspecified
  ; CHECK: Symbol table '.symtab' contains 7 entries:
  ; CHECK: TLS {{.*}} unspecified
}
