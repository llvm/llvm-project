; Test spilling of vector registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; We need to allocate a 16-byte spill slot and save the 8 call-saved FPRs.
; The frame size should be exactly 160 + 16 + 8 * 8 = 240.
define void @f1(<16 x i8> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: aghi %r15, -240
; CHECK-DAG: std %f8,
; CHECK-DAG: std %f9,
; CHECK-DAG: std %f10,
; CHECK-DAG: std %f11,
; CHECK-DAG: std %f12,
; CHECK-DAG: std %f13,
; CHECK-DAG: std %f14,
; CHECK-DAG: std %f15,
; CHECK: vst {{%v[0-9]+}}, 160(%r15)
; CHECK: vl {{%v[0-9]+}}, 160(%r15)
; CHECK-DAG: ld %f8,
; CHECK-DAG: ld %f9,
; CHECK-DAG: ld %f10,
; CHECK-DAG: ld %f11,
; CHECK-DAG: ld %f12,
; CHECK-DAG: ld %f13,
; CHECK-DAG: ld %f14,
; CHECK-DAG: ld %f15,
; CHECK: aghi %r15, 240
; CHECK: br %r14
  %v0 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v1 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v2 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v3 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v4 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v5 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v6 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v7 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v8 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v9 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v10 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v11 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v12 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v13 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v14 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v15 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v16 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v17 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v18 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v19 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v20 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v21 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v22 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v23 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v24 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v25 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v26 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v27 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v28 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v29 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v30 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v31 = load volatile <16 x i8>, <16 x i8> *%ptr
  %vx = load volatile <16 x i8>, <16 x i8> *%ptr
  store volatile <16 x i8> %vx, <16 x i8> *%ptr
  store volatile <16 x i8> %v31, <16 x i8> *%ptr
  store volatile <16 x i8> %v30, <16 x i8> *%ptr
  store volatile <16 x i8> %v29, <16 x i8> *%ptr
  store volatile <16 x i8> %v28, <16 x i8> *%ptr
  store volatile <16 x i8> %v27, <16 x i8> *%ptr
  store volatile <16 x i8> %v26, <16 x i8> *%ptr
  store volatile <16 x i8> %v25, <16 x i8> *%ptr
  store volatile <16 x i8> %v24, <16 x i8> *%ptr
  store volatile <16 x i8> %v23, <16 x i8> *%ptr
  store volatile <16 x i8> %v22, <16 x i8> *%ptr
  store volatile <16 x i8> %v21, <16 x i8> *%ptr
  store volatile <16 x i8> %v20, <16 x i8> *%ptr
  store volatile <16 x i8> %v19, <16 x i8> *%ptr
  store volatile <16 x i8> %v18, <16 x i8> *%ptr
  store volatile <16 x i8> %v17, <16 x i8> *%ptr
  store volatile <16 x i8> %v16, <16 x i8> *%ptr
  store volatile <16 x i8> %v15, <16 x i8> *%ptr
  store volatile <16 x i8> %v14, <16 x i8> *%ptr
  store volatile <16 x i8> %v13, <16 x i8> *%ptr
  store volatile <16 x i8> %v12, <16 x i8> *%ptr
  store volatile <16 x i8> %v11, <16 x i8> *%ptr
  store volatile <16 x i8> %v10, <16 x i8> *%ptr
  store volatile <16 x i8> %v9, <16 x i8> *%ptr
  store volatile <16 x i8> %v8, <16 x i8> *%ptr
  store volatile <16 x i8> %v7, <16 x i8> *%ptr
  store volatile <16 x i8> %v6, <16 x i8> *%ptr
  store volatile <16 x i8> %v5, <16 x i8> *%ptr
  store volatile <16 x i8> %v4, <16 x i8> *%ptr
  store volatile <16 x i8> %v3, <16 x i8> *%ptr
  store volatile <16 x i8> %v2, <16 x i8> *%ptr
  store volatile <16 x i8> %v1, <16 x i8> *%ptr
  store volatile <16 x i8> %v0, <16 x i8> *%ptr
  ret void
}

; Like f1, but no 16-byte slot should be needed.
define void @f2(<16 x i8> *%ptr) {
; CHECK-LABEL: f2:
; CHECK: aghi %r15, -224
; CHECK-DAG: std %f8,
; CHECK-DAG: std %f9,
; CHECK-DAG: std %f10,
; CHECK-DAG: std %f11,
; CHECK-DAG: std %f12,
; CHECK-DAG: std %f13,
; CHECK-DAG: std %f14,
; CHECK-DAG: std %f15,
; CHECK-NOT: vst {{.*}}(%r15)
; CHECK-NOT: vl {{.*}}(%r15)
; CHECK-DAG: ld %f8,
; CHECK-DAG: ld %f9,
; CHECK-DAG: ld %f10,
; CHECK-DAG: ld %f11,
; CHECK-DAG: ld %f12,
; CHECK-DAG: ld %f13,
; CHECK-DAG: ld %f14,
; CHECK-DAG: ld %f15,
; CHECK: aghi %r15, 224
; CHECK: br %r14
  %v0 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v1 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v2 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v3 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v4 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v5 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v6 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v7 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v8 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v9 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v10 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v11 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v12 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v13 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v14 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v15 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v16 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v17 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v18 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v19 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v20 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v21 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v22 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v23 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v24 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v25 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v26 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v27 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v28 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v29 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v30 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v31 = load volatile <16 x i8>, <16 x i8> *%ptr
  store volatile <16 x i8> %v31, <16 x i8> *%ptr
  store volatile <16 x i8> %v30, <16 x i8> *%ptr
  store volatile <16 x i8> %v29, <16 x i8> *%ptr
  store volatile <16 x i8> %v28, <16 x i8> *%ptr
  store volatile <16 x i8> %v27, <16 x i8> *%ptr
  store volatile <16 x i8> %v26, <16 x i8> *%ptr
  store volatile <16 x i8> %v25, <16 x i8> *%ptr
  store volatile <16 x i8> %v24, <16 x i8> *%ptr
  store volatile <16 x i8> %v23, <16 x i8> *%ptr
  store volatile <16 x i8> %v22, <16 x i8> *%ptr
  store volatile <16 x i8> %v21, <16 x i8> *%ptr
  store volatile <16 x i8> %v20, <16 x i8> *%ptr
  store volatile <16 x i8> %v19, <16 x i8> *%ptr
  store volatile <16 x i8> %v18, <16 x i8> *%ptr
  store volatile <16 x i8> %v17, <16 x i8> *%ptr
  store volatile <16 x i8> %v16, <16 x i8> *%ptr
  store volatile <16 x i8> %v15, <16 x i8> *%ptr
  store volatile <16 x i8> %v14, <16 x i8> *%ptr
  store volatile <16 x i8> %v13, <16 x i8> *%ptr
  store volatile <16 x i8> %v12, <16 x i8> *%ptr
  store volatile <16 x i8> %v11, <16 x i8> *%ptr
  store volatile <16 x i8> %v10, <16 x i8> *%ptr
  store volatile <16 x i8> %v9, <16 x i8> *%ptr
  store volatile <16 x i8> %v8, <16 x i8> *%ptr
  store volatile <16 x i8> %v7, <16 x i8> *%ptr
  store volatile <16 x i8> %v6, <16 x i8> *%ptr
  store volatile <16 x i8> %v5, <16 x i8> *%ptr
  store volatile <16 x i8> %v4, <16 x i8> *%ptr
  store volatile <16 x i8> %v3, <16 x i8> *%ptr
  store volatile <16 x i8> %v2, <16 x i8> *%ptr
  store volatile <16 x i8> %v1, <16 x i8> *%ptr
  store volatile <16 x i8> %v0, <16 x i8> *%ptr
  ret void
}

; Like f2, but only %f8 should be saved.
define void @f3(<16 x i8> *%ptr) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -168
; CHECK-DAG: std %f8,
; CHECK-NOT: vst {{.*}}(%r15)
; CHECK-NOT: vl {{.*}}(%r15)
; CHECK-NOT: %v9
; CHECK-NOT: %v10
; CHECK-NOT: %v11
; CHECK-NOT: %v12
; CHECK-NOT: %v13
; CHECK-NOT: %v14
; CHECK-NOT: %v15
; CHECK-DAG: ld %f8,
; CHECK: aghi %r15, 168
; CHECK: br %r14
  %v0 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v1 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v2 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v3 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v4 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v5 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v6 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v7 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v8 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v16 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v17 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v18 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v19 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v20 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v21 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v22 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v23 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v24 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v25 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v26 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v27 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v28 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v29 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v30 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v31 = load volatile <16 x i8>, <16 x i8> *%ptr
  store volatile <16 x i8> %v31, <16 x i8> *%ptr
  store volatile <16 x i8> %v30, <16 x i8> *%ptr
  store volatile <16 x i8> %v29, <16 x i8> *%ptr
  store volatile <16 x i8> %v28, <16 x i8> *%ptr
  store volatile <16 x i8> %v27, <16 x i8> *%ptr
  store volatile <16 x i8> %v26, <16 x i8> *%ptr
  store volatile <16 x i8> %v25, <16 x i8> *%ptr
  store volatile <16 x i8> %v24, <16 x i8> *%ptr
  store volatile <16 x i8> %v23, <16 x i8> *%ptr
  store volatile <16 x i8> %v22, <16 x i8> *%ptr
  store volatile <16 x i8> %v21, <16 x i8> *%ptr
  store volatile <16 x i8> %v20, <16 x i8> *%ptr
  store volatile <16 x i8> %v19, <16 x i8> *%ptr
  store volatile <16 x i8> %v18, <16 x i8> *%ptr
  store volatile <16 x i8> %v17, <16 x i8> *%ptr
  store volatile <16 x i8> %v16, <16 x i8> *%ptr
  store volatile <16 x i8> %v8, <16 x i8> *%ptr
  store volatile <16 x i8> %v7, <16 x i8> *%ptr
  store volatile <16 x i8> %v6, <16 x i8> *%ptr
  store volatile <16 x i8> %v5, <16 x i8> *%ptr
  store volatile <16 x i8> %v4, <16 x i8> *%ptr
  store volatile <16 x i8> %v3, <16 x i8> *%ptr
  store volatile <16 x i8> %v2, <16 x i8> *%ptr
  store volatile <16 x i8> %v1, <16 x i8> *%ptr
  store volatile <16 x i8> %v0, <16 x i8> *%ptr
  ret void
}

; Like f2, but no registers should be saved.
define void @f4(<16 x i8> *%ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r15
; CHECK: br %r14
  %v0 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v1 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v2 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v3 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v4 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v5 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v6 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v7 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v16 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v17 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v18 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v19 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v20 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v21 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v22 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v23 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v24 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v25 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v26 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v27 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v28 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v29 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v30 = load volatile <16 x i8>, <16 x i8> *%ptr
  %v31 = load volatile <16 x i8>, <16 x i8> *%ptr
  store volatile <16 x i8> %v31, <16 x i8> *%ptr
  store volatile <16 x i8> %v30, <16 x i8> *%ptr
  store volatile <16 x i8> %v29, <16 x i8> *%ptr
  store volatile <16 x i8> %v28, <16 x i8> *%ptr
  store volatile <16 x i8> %v27, <16 x i8> *%ptr
  store volatile <16 x i8> %v26, <16 x i8> *%ptr
  store volatile <16 x i8> %v25, <16 x i8> *%ptr
  store volatile <16 x i8> %v24, <16 x i8> *%ptr
  store volatile <16 x i8> %v23, <16 x i8> *%ptr
  store volatile <16 x i8> %v22, <16 x i8> *%ptr
  store volatile <16 x i8> %v21, <16 x i8> *%ptr
  store volatile <16 x i8> %v20, <16 x i8> *%ptr
  store volatile <16 x i8> %v19, <16 x i8> *%ptr
  store volatile <16 x i8> %v18, <16 x i8> *%ptr
  store volatile <16 x i8> %v17, <16 x i8> *%ptr
  store volatile <16 x i8> %v16, <16 x i8> *%ptr
  store volatile <16 x i8> %v7, <16 x i8> *%ptr
  store volatile <16 x i8> %v6, <16 x i8> *%ptr
  store volatile <16 x i8> %v5, <16 x i8> *%ptr
  store volatile <16 x i8> %v4, <16 x i8> *%ptr
  store volatile <16 x i8> %v3, <16 x i8> *%ptr
  store volatile <16 x i8> %v2, <16 x i8> *%ptr
  store volatile <16 x i8> %v1, <16 x i8> *%ptr
  store volatile <16 x i8> %v0, <16 x i8> *%ptr
  ret void
}
