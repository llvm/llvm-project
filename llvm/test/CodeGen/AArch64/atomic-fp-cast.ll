; RUN: opt -mtriple=aarch64 -S -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s --check-prefixes=FP,ALL
; RUN: opt -mtriple=aarch64 -mattr=-fp-armv8 -S -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s --check-prefixes=NOFP,ALL

define half @load_unordered_half(ptr %ptr) {
; FP-LABEL: define half @load_unordered_half(
; FP-NEXT:    [[VAL:%.*]] = load atomic half, ptr [[PTR:%.*]] unordered, align 2
; FP-NEXT:    ret half [[VAL]]
; NOFP-LABEL: define half @load_unordered_half(
; NOFP-NEXT:    [[INTVAL:%.*]] = load atomic i16, ptr [[PTR:%.*]] unordered, align 2
; NOFP-NEXT:    [[VAL:%.*]] = bitcast i16 [[INTVAL]] to half
; NOFP-NEXT:    ret half [[VAL]]
  %val = load atomic half, ptr %ptr unordered, align 2
  ret half %val
}

define bfloat @load_monotonic_bfloat(ptr %ptr) {
; FP-LABEL: define bfloat @load_monotonic_bfloat(
; FP-NEXT:    [[VAL:%.*]] = load atomic bfloat, ptr [[PTR:%.*]] monotonic, align 2
; FP-NEXT:    ret bfloat [[VAL]]
; NOFP-LABEL: define bfloat @load_monotonic_bfloat(
; NOFP-NEXT:    [[INTVAL:%.*]] = load atomic i16, ptr [[PTR:%.*]] monotonic, align 2
; NOFP-NEXT:    [[VAL:%.*]] = bitcast i16 [[INTVAL]] to bfloat
; NOFP-NEXT:    ret bfloat [[VAL]]
  %val = load atomic bfloat, ptr %ptr monotonic, align 2
  ret bfloat %val
}

define double @load_seq_cst_double(ptr %ptr) {
; ALL-LABEL: define double @load_seq_cst_double(
; ALL-NEXT:    [[INTVAL:%.*]] = load atomic i64, ptr [[PTR:%.*]] seq_cst, align 8
; ALL-NEXT:    [[VAL:%.*]] = bitcast i64 [[INTVAL]] to double
; ALL-NEXT:    ret double [[VAL]]
  %val = load atomic double, ptr %ptr seq_cst, align 8
  ret double %val
}

define void @store_unordered_half(ptr %ptr, half %val) {
; FP-LABEL: define void @store_unordered_half(
; FP-NEXT:    store atomic half [[VAL:%.*]], ptr [[PTR:%.*]] unordered, align 2
; FP-NEXT:    ret void
; NOFP-LABEL: define void @store_unordered_half(
; NOFP-NEXT:    [[INTVAL:%.*]] = bitcast half [[VAL:%.*]] to i16
; NOFP-NEXT:    store atomic i16 [[INTVAL]], ptr [[PTR:%.*]] unordered, align 2
; NOFP-NEXT:    ret void
  store atomic half %val, ptr %ptr unordered, align 2
  ret void
}

define void @store_monotonic_bfloat(ptr %ptr, bfloat %val) {
; FP-LABEL: define void @store_monotonic_bfloat(
; FP-NEXT:    store atomic bfloat [[VAL:%.*]], ptr [[PTR:%.*]] monotonic, align 2
; FP-NEXT:    ret void
; NOFP-LABEL: define void @store_monotonic_bfloat(
; NOFP-NEXT:    [[INTVAL:%.*]] = bitcast bfloat [[VAL:%.*]] to i16
; NOFP-NEXT:    store atomic i16 [[INTVAL]], ptr [[PTR:%.*]] monotonic, align 2
; NOFP-NEXT:    ret void
  store atomic bfloat %val, ptr %ptr monotonic, align 2
  ret void
}

define void @store_seq_cst_double(ptr %ptr, double %val) {
; ALL-LABEL: define void @store_seq_cst_double(
; ALL-NEXT:    [[INTVAL:%.*]] = bitcast double [[VAL:%.*]] to i64
; ALL-NEXT:    store atomic i64 [[INTVAL]], ptr [[PTR:%.*]] seq_cst, align 8
; ALL-NEXT:    ret void
  store atomic double %val, ptr %ptr seq_cst, align 8
  ret void
}
