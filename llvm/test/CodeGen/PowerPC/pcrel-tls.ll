; RUN: not --crash llc -mcpu=pwr10 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs -mattr=+pcrelative-memops -o - < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK-PCREL
; RUN: llc -mcpu=pwr10 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs -mattr=-pcrelative-memops -o - < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK-NOPCREL

; CHECK-PCREL:     Thread local storage is not supported with pc-relative addressing
; CHECK-NOPCREL:   blr

@x = external thread_local global i32, align 4

define i32* @testTLS() {
entry:
  ret i32* @x
}
