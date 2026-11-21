; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=CHECK,HLX
; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=CHECK,NOHLX

; Calls without HLX arguments or return values use the call-preserved mask of
; the callee's calling convention. With HLX disabled, hexagon_hlxcc preserves
; the same registers as the C calling convention.

; CHECK-LABEL: name: call_c
; CHECK: J2_call @c_callee, hexagoncsr,
define void @call_c() {
  call void @c_callee()
  ret void
}

; CHECK-LABEL: name: call_hlxcc
; HLX: J2_call @hlxcc_callee, hexagoncsr_hlx,
; NOHLX: J2_call @hlxcc_callee, hexagoncsr,
define void @call_hlxcc() {
  call hexagon_hlxcc void @hlxcc_callee()
  ret void
}

; CHECK-LABEL: name: call_preserve_hlx
; CHECK: J2_call @preserve_hlx_callee, hexagoncsr_preservehlx,
define void @call_preserve_hlx() {
  call hexagon_preserve_hlx void @preserve_hlx_callee()
  ret void
}

; Indirect calls use the mask of the call's calling convention.
; CHECK-LABEL: name: call_fp_hlxcc
; HLX: J2_callr {{.*}}, hexagoncsr_hlx,
; NOHLX: J2_callr {{.*}}, hexagoncsr,
define void @call_fp_hlxcc(ptr %fp) {
  call hexagon_hlxcc void %fp()
  ret void
}

; CHECK-LABEL: name: call_fp_preserve_hlx
; CHECK: J2_callr {{.*}}, hexagoncsr_preservehlx,
define void @call_fp_preserve_hlx(ptr %fp) {
  call hexagon_preserve_hlx void %fp()
  ret void
}

declare void @c_callee()
declare hexagon_hlxcc void @hlxcc_callee()
declare hexagon_preserve_hlx void @preserve_hlx_callee()
