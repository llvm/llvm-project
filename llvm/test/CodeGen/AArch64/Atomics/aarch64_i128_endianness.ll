; RUN: llc %s -o - -verify-machineinstrs -mtriple=aarch64    -mattr=+v8a -O1 | FileCheck %s
; RUN: llc %s -o - -verify-machineinstrs -mtriple=aarch64_be -mattr=+v8a -O1 | FileCheck %s

; Each function in this test performs an i128 atomic memory operation
; and also an ordinary load or store. The checks ensure that the two
; registers used in the ldxp/stxp instructions that implement the
; atomic operation appear in the same order as in the ldp/stp of the
; ordinary store. This invariant should be true in both endiannesses
; of AArch64.

define void @load_atomic(ptr %normal, ptr %atomic) {
  %value = load atomic i128, ptr %atomic monotonic, align 16
  store i128 %value, ptr %normal, align 16
  ret void
; CHECK-LABEL: load_atomic:
; CHECK: ldxp [[FIRST:x[0-9]+]], [[SECOND:x[0-9]+]], [x1]
; CHECK: stxp {{w[0-9]+}}, [[FIRST]], [[SECOND]], [x1]
; CHECK: stp [[FIRST]], [[SECOND]], [x0]
}

define void @store_atomic(ptr %normal, ptr %atomic) {
  %value = load i128, ptr %normal, align 16
  store atomic i128 %value, ptr %atomic monotonic, align 16
  ret void
; CHECK-LABEL: store_atomic:
; CHECK: ldp [[FIRST:x[0-9]+]], [[SECOND:x[0-9]+]], [x0]
; CHECK: stxp {{w[0-9]+}}, [[FIRST]], [[SECOND]], [x1]
}

define void @compare_exchange(ptr %origptr, ptr %expectedptr, ptr %newptr, ptr %atomic) {
  %new = load i128, ptr %newptr, align 16
  %expected = load i128, ptr %expectedptr, align 16
  %pair = cmpxchg ptr %atomic, i128 %expected, i128 %new monotonic monotonic, align 16
  %orig = extractvalue { i128, i1 } %pair, 0
  store i128 %orig, ptr %origptr, align 16
  ret void
; CHECK-LABEL: compare_exchange:
; CHECK-DAG: ldp [[NEWFIRST:x[0-9]+]], [[NEWSECOND:x[0-9]+]], [x2]
; CHECK-DAG: ldp [[EXPFIRST:x[0-9]+]], [[EXPSECOND:x[0-9]+]], [x1]
; CHECK: ldxp [[ATOMICFIRST:x[0-9]+]], [[ATOMICSECOND:x[0-9]+]], [x3]
; CHECK-DAG: cmp [[ATOMICFIRST]], [[EXPFIRST]]
; CHECK-DAG: cmp [[ATOMICSECOND]], [[EXPSECOND]]
; CHECK: stxp {{w[0-9]+}}, [[ATOMICFIRST]], [[ATOMICSECOND]], [x3]
; CHECK: stxp {{w[0-9]+}}, [[NEWFIRST]], [[NEWSECOND]], [x3]
; CHECK: stp [[ATOMICFIRST]], [[ATOMICSECOND]], [x0]
}
