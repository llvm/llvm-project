; RUN: llc < %s -mcpu=sm_90 -mattr=+ptx78 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}

; PTX .ftz flushes subnormal inputs *and* results, so it implements only the
; preservesign|preservesign mode. Flushing an output is permitted but never
; mandated, while flushing an ieee input is forbidden; conversely a preservesign
; input mode mandates a flush that a plain op does not perform.

target triple = "nvptx64-nvidia-cuda"

declare float @llvm.canonicalize.f32(float)

; The output permission alone must not buy .ftz: the inputs are ieee.
define float @fadd_preservesign_ieee(float %x, float %y) #0 {
; CHECK-LABEL: fadd_preservesign_ieee(
; CHECK:    add.rn.f32
; CHECK-NOT:    add.rn.ftz.f32
  %r = fadd float %x, %y
  ret float %r
}

; Flushing the result is not permitted here, so .ftz is out.
define float @fadd_ieee_preservesign(float %x, float %y) #1 {
; CHECK-LABEL: fadd_ieee_preservesign(
; CHECK:    add.rn.f32
; CHECK-NOT:    add.rn.ftz.f32
  %r = fadd float %x, %y
  ret float %r
}

; The one mode .ftz actually implements. This is what a command line produces.
define float @fadd_preservesign_both(float %x, float %y) #2 {
; CHECK-LABEL: fadd_preservesign_both(
; CHECK:    add.rn.ftz.f32
  %r = fadd float %x, %y
  ret float %r
}

define float @fadd_ieee_both(float %x, float %y) #3 {
; CHECK-LABEL: fadd_ieee_both(
; CHECK:    add.rn.f32
; CHECK-NOT:    add.rn.ftz.f32
  %r = fadd float %x, %y
  ret float %r
}

; A mandated input flush that no plain op performs: canonicalize has to carry
; .ftz even though the output mode forbids flushing results.
define float @canonicalize_ieee_preservesign(float %x) #1 {
; CHECK-LABEL: canonicalize_ieee_preservesign(
; CHECK:    mul.rn.ftz.f32
  %r = call float @llvm.canonicalize.f32(float %x)
  ret float %r
}

; Inputs are ieee, so canonicalize must not flush them.
define float @canonicalize_preservesign_ieee(float %x) #0 {
; CHECK-LABEL: canonicalize_preservesign_ieee(
; CHECK:    mul.rn.f32
; CHECK-NOT:    mul.rn.ftz.f32
  %r = call float @llvm.canonicalize.f32(float %x)
  ret float %r
}

define float @canonicalize_preservesign_both(float %x) #2 {
; CHECK-LABEL: canonicalize_preservesign_both(
; CHECK:    mul.rn.ftz.f32
  %r = call float @llvm.canonicalize.f32(float %x)
  ret float %r
}

define float @canonicalize_ieee_both(float %x) #3 {
; CHECK-LABEL: canonicalize_ieee_both(
; CHECK:    mul.rn.f32
; CHECK-NOT:    mul.rn.ftz.f32
  %r = call float @llvm.canonicalize.f32(float %x)
  ret float %r
}

attributes #0 = { denormal_fpenv(float: preservesign|ieee) }
attributes #1 = { denormal_fpenv(float: ieee|preservesign) }
attributes #2 = { denormal_fpenv(float: preservesign|preservesign) }
attributes #3 = { denormal_fpenv(float: ieee|ieee) }
