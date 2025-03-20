; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+lse2 %s -o - | FileCheck %s

define dso_local half @load_atomic_f16_aligned_unordered(ptr %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_unordered:
; CHECK:    ldrh w8, [x0]
    %r = load atomic half, ptr %ptr unordered, align 2
    ret half %r
}

define dso_local half @load_atomic_f16_aligned_unordered_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_unordered_const:
; CHECK:    ldrh w8, [x0]
    %r = load atomic half, ptr %ptr unordered, align 2
    ret half %r
}

define dso_local half @load_atomic_f16_aligned_monotonic(ptr %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_monotonic:
; CHECK:    ldrh w8, [x0]
    %r = load atomic half, ptr %ptr monotonic, align 2
    ret half %r
}

define dso_local half @load_atomic_f16_aligned_monotonic_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_monotonic_const:
; CHECK:    ldrh w8, [x0]
    %r = load atomic half, ptr %ptr monotonic, align 2
    ret half %r
}

define dso_local half @load_atomic_f16_aligned_acquire(ptr %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_acquire:
; CHECK:    ldarh w8, [x0]
    %r = load atomic half, ptr %ptr acquire, align 2
    ret half %r
}

define dso_local half @load_atomic_f16_aligned_acquire_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_acquire_const:
; CHECK:    ldarh w8, [x0]
    %r = load atomic half, ptr %ptr acquire, align 2
    ret half %r
}

define dso_local half @load_atomic_f16_aligned_seq_cst(ptr %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_seq_cst:
; CHECK:    ldarh w8, [x0]
    %r = load atomic half, ptr %ptr seq_cst, align 2
    ret half %r
}

define dso_local half @load_atomic_f16_aligned_seq_cst_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_f16_aligned_seq_cst_const:
; CHECK:    ldarh w8, [x0]
    %r = load atomic half, ptr %ptr seq_cst, align 2
    ret half %r
}

define dso_local bfloat @load_atomic_bf16_aligned_unordered(ptr %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_unordered:
; CHECK:    ldrh w8, [x0]
    %r = load atomic bfloat, ptr %ptr unordered, align 2
    ret bfloat %r
}

define dso_local bfloat @load_atomic_bf16_aligned_unordered_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_unordered_const:
; CHECK:    ldrh w8, [x0]
    %r = load atomic bfloat, ptr %ptr unordered, align 2
    ret bfloat %r
}

define dso_local bfloat @load_atomic_bf16_aligned_monotonic(ptr %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_monotonic:
; CHECK:    ldrh w8, [x0]
    %r = load atomic bfloat, ptr %ptr monotonic, align 2
    ret bfloat %r
}

define dso_local bfloat @load_atomic_bf16_aligned_monotonic_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_monotonic_const:
; CHECK:    ldrh w8, [x0]
    %r = load atomic bfloat, ptr %ptr monotonic, align 2
    ret bfloat %r
}

define dso_local bfloat @load_atomic_bf16_aligned_acquire(ptr %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_acquire:
; CHECK:    ldarh w8, [x0]
    %r = load atomic bfloat, ptr %ptr acquire, align 2
    ret bfloat %r
}

define dso_local bfloat @load_atomic_bf16_aligned_acquire_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_acquire_const:
; CHECK:    ldarh w8, [x0]
    %r = load atomic bfloat, ptr %ptr acquire, align 2
    ret bfloat %r
}

define dso_local bfloat @load_atomic_bf16_aligned_seq_cst(ptr %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_seq_cst:
; CHECK:    ldarh w8, [x0]
    %r = load atomic bfloat, ptr %ptr seq_cst, align 2
    ret bfloat %r
}

define dso_local bfloat @load_atomic_bf16_aligned_seq_cst_const(ptr readonly %ptr) {
; CHECK-LABEL: load_atomic_bf16_aligned_seq_cst_const:
; CHECK:    ldarh w8, [x0]
    %r = load atomic bfloat, ptr %ptr seq_cst, align 2
    ret bfloat %r
}
