; RUN: llc -mtriple=aarch64 -verify-machineinstrs < %s | \
; RUN:   FileCheck %s --check-prefix=NO-RCPC3
; RUN: llc -mtriple=aarch64 -mattr=+rcpc3 -verify-machineinstrs < %s | \
; RUN:   FileCheck %s --check-prefix=RCPC3
; RUN: llc -mtriple=aarch64 -mattr=-fp-armv8 -verify-machineinstrs < %s | \
; RUN:   FileCheck %s --check-prefix=NOFP

define half @load_acquire_half(ptr %ptr) {
; NO-RCPC3-LABEL: load_acquire_half:
; NO-RCPC3:       // %bb.0:
; NO-RCPC3-NEXT:    ldarh w8, [x0]
; NO-RCPC3-NEXT:    fmov s0, w8
; NO-RCPC3:         ret
; RCPC3-LABEL: load_acquire_half:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur h0, [x0]
; RCPC3-NEXT:    ret
; NOFP-LABEL: load_acquire_half:
; NOFP:       // %bb.0:
; NOFP-NEXT:    ldarh w0, [x0]
; NOFP-NEXT:    ret
  %val = load atomic half, ptr %ptr acquire, align 2
  ret half %val
}

define half @load_acquire_half_offset(ptr %ptr) {
; RCPC3-LABEL: load_acquire_half_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur h0, [x0, #-2]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -2
  %val = load atomic half, ptr %addr acquire, align 2
  ret half %val
}

define bfloat @load_acquire_bfloat(ptr %ptr) {
; NO-RCPC3-LABEL: load_acquire_bfloat:
; NO-RCPC3:       // %bb.0:
; NO-RCPC3-NEXT:    ldarh w8, [x0]
; NO-RCPC3-NEXT:    fmov s0, w8
; NO-RCPC3:         ret
; RCPC3-LABEL: load_acquire_bfloat:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur h0, [x0]
; RCPC3-NEXT:    ret
; NOFP-LABEL: load_acquire_bfloat:
; NOFP:       // %bb.0:
; NOFP-NEXT:    ldarh w0, [x0]
; NOFP-NEXT:    ret
  %val = load atomic bfloat, ptr %ptr acquire, align 2
  ret bfloat %val
}

define bfloat @load_acquire_bfloat_offset(ptr %ptr) {
; RCPC3-LABEL: load_acquire_bfloat_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur h0, [x0, #-2]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -2
  %val = load atomic bfloat, ptr %addr acquire, align 2
  ret bfloat %val
}

define float @load_acquire_float(ptr %ptr) {
; NO-RCPC3-LABEL: load_acquire_float:
; NO-RCPC3:       // %bb.0:
; NO-RCPC3-NEXT:    ldar w8, [x0]
; NO-RCPC3-NEXT:    fmov s0, w8
; NO-RCPC3-NEXT:    ret
; RCPC3-LABEL: load_acquire_float:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur s0, [x0]
; RCPC3-NEXT:    ret
; NOFP-LABEL: load_acquire_float:
; NOFP:       // %bb.0:
; NOFP-NEXT:    ldar w0, [x0]
; NOFP-NEXT:    ret
  %val = load atomic float, ptr %ptr acquire, align 4
  ret float %val
}

define float @load_acquire_float_offset(ptr %ptr) {
; RCPC3-LABEL: load_acquire_float_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur s0, [x0, #-4]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -4
  %val = load atomic float, ptr %addr acquire, align 4
  ret float %val
}

define double @load_acquire_double(ptr %ptr) {
; NO-RCPC3-LABEL: load_acquire_double:
; NO-RCPC3:       // %bb.0:
; NO-RCPC3-NEXT:    ldar x8, [x0]
; NO-RCPC3-NEXT:    fmov d0, x8
; NO-RCPC3-NEXT:    ret
; RCPC3-LABEL: load_acquire_double:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur d0, [x0]
; RCPC3-NEXT:    ret
; NOFP-LABEL: load_acquire_double:
; NOFP:       // %bb.0:
; NOFP-NEXT:    ldar x0, [x0]
; NOFP-NEXT:    ret
  %val = load atomic double, ptr %ptr acquire, align 8
  ret double %val
}

define double @load_acquire_double_offset(ptr %ptr) {
; RCPC3-LABEL: load_acquire_double_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    ldapur d0, [x0, #-8]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -8
  %val = load atomic double, ptr %addr acquire, align 8
  ret double %val
}
