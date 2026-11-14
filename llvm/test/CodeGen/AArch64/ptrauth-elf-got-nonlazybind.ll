; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+pauth -global-isel=0 -fast-isel=0 \
; RUN:   -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+pauth -global-isel=0 -fast-isel=1 \
; RUN:   -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+pauth -global-isel=1 -global-isel-abort=1 \
; RUN:   -verify-machineinstrs %s -o - | FileCheck %s

;; With -fno-plt, a call to a nonlazybind function loads the callee from the
;; GOT. With a signed GOT, the loaded pointer has to be authenticated before
;; the call. FastISel falls back to SelectionDAG for calls when RtLibUseGOT is
;; set.

declare i32 @callee() nonlazybind

define i32 @call_nonlazybind() {
; CHECK-LABEL: call_nonlazybind:
; CHECK:         adrp  x17, :got_auth:callee
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:callee
; CHECK-NEXT:    ldr   x16, [x17]
; CHECK-NEXT:    autia x16, x17
; CHECK-NEXT:    mov   x17, x16
; CHECK-NEXT:    xpaci x17
; CHECK-NEXT:    cmp   x16, x17
; CHECK-NEXT:    b.eq  [[L0:.Lauth_success_[0-9]+]]
; CHECK-NEXT:    brk   #0xc470
; CHECK-NEXT:  [[L0]]:
; CHECK-NEXT:    mov   [[R0:x[0-9]+]], x16
; CHECK-NEXT:    blr   [[R0]]
  %r = call i32 @callee()
  %a = add i32 %r, 1
  ret i32 %a
}

define i32 @tail_call_nonlazybind() {
; CHECK-LABEL: tail_call_nonlazybind:
; CHECK:         adrp  x17, :got_auth:callee
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:callee
; CHECK-NEXT:    ldr   x16, [x17]
; CHECK-NEXT:    autia x16, x17
; CHECK-NEXT:    mov   x17, x16
; CHECK-NEXT:    xpaci x17
; CHECK-NEXT:    cmp   x16, x17
; CHECK-NEXT:    b.eq  [[L1:.Lauth_success_[0-9]+]]
; CHECK-NEXT:    brk   #0xc470
; CHECK-NEXT:  [[L1]]:
; CHECK-NEXT:    mov   [[R1:x[0-9]+]], x16
; CHECK-NEXT:    br    [[R1]]
  %r = tail call i32 @callee()
  ret i32 %r
}

;; The linker signs the GOT slot with the IA key only for an STT_FUNC symbol.
; CHECK: .type callee,@function

!llvm.module.flags = !{!0, !1}
!0 = !{i32 7, !"RtLibUseGOT", i32 1}
!1 = !{i32 8, !"ptrauth-elf-got", i32 1}
