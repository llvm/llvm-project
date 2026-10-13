; RUN: llc -mtriple=aarch64 -mattr=+rcpc3 -verify-machineinstrs < %s | \
; RUN:   FileCheck %s --check-prefix=RCPC3
; RUN: llc -mtriple=aarch64 -verify-machineinstrs < %s | \
; RUN:   FileCheck %s --check-prefix=NO-RCPC3
; RUN: llc -mtriple=aarch64 -mattr=-fp-armv8 -verify-machineinstrs < %s | \
; RUN:   FileCheck %s --check-prefix=NOFP

define void @store_release_half(ptr %ptr, half %val) nounwind {
; RCPC3-LABEL: store_release_half:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur h0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_release_half:
; NO-RCPC3:       fmov w8, s0
; NO-RCPC3-NEXT:  stlrh w8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_release_half:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlrh w1, [x0]
; NOFP-NEXT:    ret
  store atomic half %val, ptr %ptr release, align 2
  ret void
}

define void @store_release_half_offset(ptr %ptr, half %val) nounwind {
; RCPC3-LABEL: store_release_half_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur h0, [x0, #-2]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -2
  store atomic half %val, ptr %addr release, align 2
  ret void
}

define void @store_release_bfloat(ptr %ptr, bfloat %val) nounwind {
; RCPC3-LABEL: store_release_bfloat:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur h0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_release_bfloat:
; NO-RCPC3:       fmov w8, s0
; NO-RCPC3-NEXT:  stlrh w8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_release_bfloat:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlrh w1, [x0]
; NOFP-NEXT:    ret
  store atomic bfloat %val, ptr %ptr release, align 8
  ret void
}

define void @store_release_bfloat_offset(ptr %ptr, bfloat %val) nounwind {
; RCPC3-LABEL: store_release_bfloat_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur h0, [x0, #-2]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -2
  store atomic bfloat %val, ptr %addr release, align 2
  ret void
}

define void @store_release_float(ptr %ptr, float %val) nounwind {
; RCPC3-LABEL: store_release_float:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur s0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_release_float:
; NO-RCPC3:       fmov w8, s0
; NO-RCPC3-NEXT:  stlr w8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_release_float:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlr w1, [x0]
; NOFP-NEXT:    ret
  store atomic float %val, ptr %ptr release, align 4
  ret void
}

define void @store_release_float_offset(ptr %ptr, float %val) nounwind {
; RCPC3-LABEL: store_release_float_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur s0, [x0, #-4]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -4
  store atomic float %val, ptr %addr release, align 4
  ret void
}

define void @store_release_double(ptr %ptr, double %val) nounwind {
; RCPC3-LABEL: store_release_double:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur d0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_release_double:
; NO-RCPC3:       fmov x8, d0
; NO-RCPC3-NEXT:  stlr x8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_release_double:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlr x1, [x0]
; NOFP-NEXT:    ret
  store atomic double %val, ptr %ptr release, align 8
  ret void
}

define void @store_release_double_offset(ptr %ptr, double %val) nounwind {
; RCPC3-LABEL: store_release_double_offset:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur d0, [x0, #-8]
; RCPC3-NEXT:    ret
  %addr = getelementptr i8, ptr %ptr, i64 -8
  store atomic double %val, ptr %addr release, align 8
  ret void
}

define void @store_seq_cst_half(ptr %ptr, half %val) nounwind {
; RCPC3-LABEL: store_seq_cst_half:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur h0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_seq_cst_half:
; NO-RCPC3:       fmov w8, s0
; NO-RCPC3-NEXT:  stlrh w8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_seq_cst_half:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlrh w1, [x0]
; NOFP-NEXT:    ret
  store atomic half %val, ptr %ptr seq_cst, align 2
  ret void
}

define void @store_seq_cst_bfloat(ptr %ptr, bfloat %val) nounwind {
; RCPC3-LABEL: store_seq_cst_bfloat:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur h0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_seq_cst_bfloat:
; NO-RCPC3:       fmov w8, s0
; NO-RCPC3-NEXT:  stlrh w8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_seq_cst_bfloat:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlrh w1, [x0]
; NOFP-NEXT:    ret
  store atomic bfloat %val, ptr %ptr seq_cst, align 2
  ret void
}

define void @store_seq_cst_float(ptr %ptr, float %val) nounwind {
; RCPC3-LABEL: store_seq_cst_float:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur s0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_seq_cst_float:
; NO-RCPC3:       fmov w8, s0
; NO-RCPC3-NEXT:  stlr w8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_seq_cst_float:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlr w1, [x0]
; NOFP-NEXT:    ret
  store atomic float %val, ptr %ptr seq_cst, align 4
  ret void
}

define void @store_seq_cst_double(ptr %ptr, double %val) nounwind {
; RCPC3-LABEL: store_seq_cst_double:
; RCPC3:       // %bb.0:
; RCPC3-NEXT:    stlur d0, [x0]
; RCPC3-NEXT:    ret
; NO-RCPC3-LABEL: store_seq_cst_double:
; NO-RCPC3:       fmov x8, d0
; NO-RCPC3-NEXT:  stlr x8, [x0]
; NO-RCPC3-NEXT:  ret
; NOFP-LABEL: store_seq_cst_double:
; NOFP:       // %bb.0:
; NOFP-NEXT:    stlr x1, [x0]
; NOFP-NEXT:    ret
  store atomic double %val, ptr %ptr seq_cst, align 8
  ret void
}
