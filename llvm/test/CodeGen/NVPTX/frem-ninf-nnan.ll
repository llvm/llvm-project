; RUN: llc %s --stop-after=finalize-isel -mcpu=sm_60 -o - | FileCheck %s

target triple = "nvptx64-unknown-cuda"

define float @frem_ninf_nnan(float %a, float %b) {
  ; CHECK:     nnan ninf FDIV32rr_prec
  ; CHECK-NOT: nnan ninf contract FNEGf32
  ; CHECK:     contract FNEGf32
  %r = frem ninf nnan float %a, %b
  ret float %r
}
