; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=volatile --test FileCheck --test-arg --check-prefixes=INTERESTING,CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,CHECK %s < %t

; CHECK-LABEL: @load_volatile_keep(
; INTERESTING: load volatile
; RESULT: %op = load volatile i32,
define i32 @load_volatile_keep(ptr %ptr) {
  %op = load volatile i32, ptr %ptr
  ret i32 %op
}

; CHECK-LABEL: @load_volatile_drop(
; INTERESTING: load
; RESULT: %op = load i32,
define i32 @load_volatile_drop(ptr %ptr) {
  %op = load volatile i32, ptr %ptr
  ret i32 %op
}

; CHECK-LABEL: @store_volatile_keep(
; INTERESTING: store volatile
; RESULT: store volatile i32 0,
define void @store_volatile_keep(ptr %ptr) {
  store volatile i32 0, ptr %ptr
  ret void
}

; CHECK-LABEL: @store_volatile_drop(
; INTERESTING: store
; RESULT: store i32 0,
define void @store_volatile_drop(ptr %ptr) {
  store volatile i32 0, ptr %ptr
  ret void
}

; CHECK-LABEL: @atomicrmw_volatile_keep(
; INTERESTING: atomicrmw volatile
; RESULT: atomicrmw volatile add ptr %ptr
define i32 @atomicrmw_volatile_keep(ptr %ptr) {
  %val = atomicrmw volatile add ptr %ptr, i32 3 seq_cst
  ret i32 %val
}

; CHECK-LABEL: @atomicrmw_volatile_drop(
; INTERESTING: atomicrmw
; RESULT: atomicrmw add ptr %ptr
define i32 @atomicrmw_volatile_drop(ptr %ptr) {
  %val = atomicrmw volatile add ptr %ptr, i32 3 seq_cst
  ret i32 %val
}

; CHECK-LABEL: @cmpxchg_volatile_keep(
; INTERESTING: cmpxchg volatile
; RESULT: cmpxchg volatile ptr %ptr, i32 %old, i32 %in seq_cst seq_cst
define { i32, i1 } @cmpxchg_volatile_keep(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg volatile ptr %ptr, i32 %old, i32 %in seq_cst seq_cst
  ret { i32, i1 } %val
}

; CHECK-LABEL: @cmpxchg_volatile_drop(
; INTERESTING: cmpxchg
; RESULT: cmpxchg ptr %ptr, i32 %old, i32 %in seq_cst seq_cst
define { i32, i1 } @cmpxchg_volatile_drop(ptr %ptr, i32 %old, i32 %in) {
  %val = cmpxchg volatile ptr %ptr, i32 %old, i32 %in seq_cst seq_cst
  ret { i32, i1 } %val
}

; CHECK-LABEL: @memcpy_volatile_keep(
; INTERESTING: i1 true
; RESULT: call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %size, i1 true)
define void @memcpy_volatile_keep(ptr %dst, ptr %src, i64 %size) {
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %size, i1 true)
  ret void
}

; CHECK-LABEL: @memcpy_volatile_drop(
; INTERESTING: llvm.memcpy
; RESULT: call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %size, i1 false)
define void @memcpy_volatile_drop(ptr %dst, ptr %src, i64 %size) {
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %size, i1 true)
  ret void
}

; CHECK-LABEL: @memcpy_inline_volatile_keep(
; INTERESTING: i1 true
; RESULT: call void @llvm.memcpy.inline.p0.p0.i64(ptr %dst, ptr %src, i64 256, i1 true)
define void @memcpy_inline_volatile_keep(ptr %dst, ptr %src) {
  call void @llvm.memcpy.inline.p0.p0.i64(ptr %dst, ptr %src, i64 256, i1 true)
  ret void
}

; CHECK-LABEL: @memcpy_inline_volatile_drop(
; INTERESTING: llvm.memcpy
; RESULT: call void @llvm.memcpy.inline.p0.p0.i64(ptr %dst, ptr %src, i64 256, i1 false)
define void @memcpy_inline_volatile_drop(ptr %dst, ptr %src) {
  call void @llvm.memcpy.inline.p0.p0.i64(ptr %dst, ptr %src, i64 256, i1 true)
  ret void
}

; CHECK-LABEL: @memmove_volatile_keep(
; INTERESTING: i1 true
; RESULT: call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 256, i1 true)
define void @memmove_volatile_keep(ptr %dst, ptr %src) {
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 256, i1 true)
  ret void
}

; CHECK-LABEL: @memmove_volatile_drop(
; INTERESTING: llvm.memmove
; RESULT: call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %size, i1 false)
define void @memmove_volatile_drop(ptr %dst, ptr %src, i64 %size) {
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %size, i1 true)
  ret void
}

; CHECK-LABEL: @memset_volatile_keep(
; INTERESTING: i1 true
; RESULT: call void @llvm.memset.p0.i64(ptr %ptr, i8 %val, i64 %size, i1 true)
define void @memset_volatile_keep(ptr %ptr, i8 %val, i64 %size) {
  call void @llvm.memset.p0.i64(ptr %ptr, i8 %val, i64 %size, i1 true)
  ret void
}

; CHECK-LABEL: @memset_volatile_drop(
; INTERESTING: llvm.memset
; RESULT: call void @llvm.memset.p0.i64(ptr %ptr, i8 %val, i64 %size, i1 false)
define void @memset_volatile_drop(ptr %ptr, i8 %val, i64 %size) {
  call void @llvm.memset.p0.i64(ptr %ptr, i8 %val, i64 %size, i1 true)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.inline.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64 immarg, i1 immarg)
declare void @llvm.memset.p0.i64(ptr noalias nocapture readonly, i8, i64, i1 immarg)
