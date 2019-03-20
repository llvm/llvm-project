; RUN: llc -O0 -fast-isel -verify-machineinstrs -mtriple=arm64-eabi < %s | FileCheck --enable-var-scope %s

; Test fptosi
define i32 @fptosi_wh(half %a) nounwind ssp {
entry:
; CHECK-LABEL: fptosi_wh
; CHECK: fcvt s1, h0
; CHECK: fcvtzs [[REG:w[0-9]+]], s1
; CHECK: mov w0, [[REG]]
  %conv = fptosi half %a to i32
  ret i32 %conv
}

; Test fptoui
define i32 @fptoui_swh(half %a) nounwind ssp {
entry:
; CHECK-LABEL: fptoui_swh
; CHECK: fcvt s1, h0
; CHECK: fcvtzu [[REG:w[0-9]+]], s1
; CHECK: mov w0, [[REG]]
  %conv = fptoui half %a to i32
  ret i32 %conv
}

; Test sitofp
define half @sitofp_hw_i1(i1 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_hw_i1
; CHECK: sbfx w8, w0, #0, #1
; CHECK: scvtf s0, w8
; CHECK: fcvt  h0, s0
  %conv = sitofp i1 %a to half
  ret half %conv
}

; Test sitofp
define half @sitofp_hw_i8(i8 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_hw_i8
; CHECK: sxtb w8, w0
; CHECK: scvtf s0, w8
; CHECK: fcvt  h0, s0
  %conv = sitofp i8 %a to half
  ret half %conv
}

; Test sitofp
define half @sitofp_hw_i16(i16 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_hw_i16
; CHECK: sxth w8, w0
; CHECK: scvtf s0, w8
; CHECK: fcvt  h0, s0
  %conv = sitofp i16 %a to half
  ret half %conv
}

; Test sitofp
define half @sitofp_hw_i32(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_hw_i32
; CHECK: scvtf s0, w0
; CHECK: fcvt  h0, s0
  %conv = sitofp i32 %a to half
  ret half %conv
}

; Test sitofp
define half @sitofp_hx(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_hx
; CHECK: scvtf s0, x0
; CHECK: fcvt  h0, s0
  %conv = sitofp i64 %a to half
  ret half %conv
}

; Test uitofp
define half @uitofp_hw_i1(i1 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_hw_i1
; CHECK: and w8, w0, #0x1
; CHECK: ucvtf s0, w8
; CHECK: fcvt  h0, s0
  %conv = uitofp i1 %a to half
  ret half %conv
}

; Test uitofp
define half @uitofp_hw_i8(i8 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_hw_i8
; CHECK: and w8, w0, #0xff
; CHECK: ucvtf s0, w8
; CHECK: fcvt  h0, s0
  %conv = uitofp i8 %a to half
  ret half %conv
}

; Test uitofp
define half @uitofp_hw_i16(i16 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_hw_i16
; CHECK: and w8, w0, #0xffff
; CHECK: ucvtf s0, w8
; CHECK: fcvt  h0, s0
  %conv = uitofp i16 %a to half
  ret half %conv
}

; Test uitofp
define half @uitofp_hw_i32(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_hw_i32
; CHECK: ucvtf s0, w0
; CHECK: fcvt  h0, s0
  %conv = uitofp i32 %a to half
  ret half %conv
}

; Test uitofp
define half @uitofp_hx(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_hx
; CHECK: ucvtf s0, x0
; CHECK: fcvt  h0, s0
  %conv = uitofp i64 %a to half
  ret half %conv
}


