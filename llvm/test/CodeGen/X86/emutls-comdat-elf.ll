; RUN: llc < %s -emulated-tls -mtriple=x86_64-linux-gnu \
; RUN:   | FileCheck %s --implicit-check-not=",S,comdat"

; Members of a COMDAT keyed by an emulated thread_local join the group of
; __emutls_v.S, which replaces S as the key.

$S = comdat any

@S = linkonce_odr thread_local global i32 7, comdat, align 4

define linkonce_odr ptr @S_member() comdat($S) {
  ret ptr @S
}

; CHECK: .section .text.S_member,"axG",@progbits,__emutls_v.S,comdat
; CHECK: .section .data.__emutls_v.S,"awG",@progbits,__emutls_v.S,comdat
