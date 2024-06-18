; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1              -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth %s -o - | FileCheck %s

;; Note: for FastISel, we fall back to SelectionDAG

@var = global i32 0

define i32 @get_globalvar() {
; CHECK-LABEL: get_globalvar:

  %val = load i32, ptr @var

; CHECK: adrp x[[GOT:[0-9]+]], :got_auth:var
; CHECK: add x[[GOT]], x[[GOT]], :got_auth_lo12:var
; CHECK: ldr x[[SYM:[0-9]+]], [x[[GOT]]]
; CHECK: autda x[[SYM]], x[[GOT]]
; CHECK: ldr w0, [x[[SYM]]]
  ret i32 %val
}

define ptr @get_globalvaraddr() {
; CHECK-LABEL: get_globalvaraddr:

  %val = load i32, ptr @var

; CHECK: adrp x[[GOT:[0-9]+]], :got_auth:var
; CHECK: add x[[GOT]], x[[GOT]], :got_auth_lo12:var
; CHECK: ldr x0, [x[[GOT]]]
; CHECK: autda x0, x[[GOT]]
  ret ptr @var
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 128}
