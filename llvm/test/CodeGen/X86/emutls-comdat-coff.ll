; RUN: llc < %s -emulated-tls -mtriple=x86_64-pc-windows-msvc \
; RUN:   | FileCheck %s --implicit-check-not="associative,S"
; RUN: llc < %s -emulated-tls -mtriple=x86_64-pc-windows-msvc -filetype=obj \
; RUN:   -o /dev/null
; RUN: opt < %s -passes=lower-emutls -S | FileCheck %s --check-prefix=IR

; `inline thread_local T S;` in the MSVC ABI: the .CRT$XDU entry is
; associative to S's COMDAT. With emulated TLS, __emutls_v.S becomes the key.

$S = comdat any

@S = linkonce_odr thread_local global i32 7, comdat, align 4
@S_init_entry = internal constant ptr @S_init, section ".CRT$XDU", comdat($S)

define internal void @S_init() {
  ret void
}

define linkonce_odr ptr @S_member() comdat($S) {
  ret ptr @S
}

; A COMDAT keyed by a function keeps its key.
$f = comdat any

@tf = linkonce_odr thread_local global i32 1, comdat($f), align 4

define linkonce_odr ptr @f() comdat {
  ret ptr @tf
}

; CHECK: .section .text,"xr",associative,__emutls_v.S
; CHECK: S_member:
; CHECK: .section .text,"xr",discard,f
; CHECK: .section .CRT$XDU,"dr",associative,__emutls_v.S
; CHECK: .section .data,"dw",discard,__emutls_v.S

; IR: @S_init_entry = {{.*}} comdat($__emutls_v.S)
; IR: define linkonce_odr ptr @S_member() comdat($__emutls_v.S)
; IR: define linkonce_odr ptr @f() comdat {
