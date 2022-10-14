; Test spilling of vector registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; We need to allocate a 16-byte spill slot and save the 8 call-saved FPRs.
; The frame size should be exactly 160 + 16 + 8 * 8 = 240.
define void @f1(ptr %ptr) {
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
; CHECK: vst {{%v[0-9]+}}, 160(%r15), 3
; CHECK: vl {{%v[0-9]+}}, 160(%r15), 3
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
  %v0 = load volatile <16 x i8>, ptr %ptr
  %v1 = load volatile <16 x i8>, ptr %ptr
  %v2 = load volatile <16 x i8>, ptr %ptr
  %v3 = load volatile <16 x i8>, ptr %ptr
  %v4 = load volatile <16 x i8>, ptr %ptr
  %v5 = load volatile <16 x i8>, ptr %ptr
  %v6 = load volatile <16 x i8>, ptr %ptr
  %v7 = load volatile <16 x i8>, ptr %ptr
  %v8 = load volatile <16 x i8>, ptr %ptr
  %v9 = load volatile <16 x i8>, ptr %ptr
  %v10 = load volatile <16 x i8>, ptr %ptr
  %v11 = load volatile <16 x i8>, ptr %ptr
  %v12 = load volatile <16 x i8>, ptr %ptr
  %v13 = load volatile <16 x i8>, ptr %ptr
  %v14 = load volatile <16 x i8>, ptr %ptr
  %v15 = load volatile <16 x i8>, ptr %ptr
  %v16 = load volatile <16 x i8>, ptr %ptr
  %v17 = load volatile <16 x i8>, ptr %ptr
  %v18 = load volatile <16 x i8>, ptr %ptr
  %v19 = load volatile <16 x i8>, ptr %ptr
  %v20 = load volatile <16 x i8>, ptr %ptr
  %v21 = load volatile <16 x i8>, ptr %ptr
  %v22 = load volatile <16 x i8>, ptr %ptr
  %v23 = load volatile <16 x i8>, ptr %ptr
  %v24 = load volatile <16 x i8>, ptr %ptr
  %v25 = load volatile <16 x i8>, ptr %ptr
  %v26 = load volatile <16 x i8>, ptr %ptr
  %v27 = load volatile <16 x i8>, ptr %ptr
  %v28 = load volatile <16 x i8>, ptr %ptr
  %v29 = load volatile <16 x i8>, ptr %ptr
  %v30 = load volatile <16 x i8>, ptr %ptr
  %v31 = load volatile <16 x i8>, ptr %ptr
  %vx = load volatile <16 x i8>, ptr %ptr
  store volatile <16 x i8> %vx, ptr %ptr
  store volatile <16 x i8> %v31, ptr %ptr
  store volatile <16 x i8> %v30, ptr %ptr
  store volatile <16 x i8> %v29, ptr %ptr
  store volatile <16 x i8> %v28, ptr %ptr
  store volatile <16 x i8> %v27, ptr %ptr
  store volatile <16 x i8> %v26, ptr %ptr
  store volatile <16 x i8> %v25, ptr %ptr
  store volatile <16 x i8> %v24, ptr %ptr
  store volatile <16 x i8> %v23, ptr %ptr
  store volatile <16 x i8> %v22, ptr %ptr
  store volatile <16 x i8> %v21, ptr %ptr
  store volatile <16 x i8> %v20, ptr %ptr
  store volatile <16 x i8> %v19, ptr %ptr
  store volatile <16 x i8> %v18, ptr %ptr
  store volatile <16 x i8> %v17, ptr %ptr
  store volatile <16 x i8> %v16, ptr %ptr
  store volatile <16 x i8> %v15, ptr %ptr
  store volatile <16 x i8> %v14, ptr %ptr
  store volatile <16 x i8> %v13, ptr %ptr
  store volatile <16 x i8> %v12, ptr %ptr
  store volatile <16 x i8> %v11, ptr %ptr
  store volatile <16 x i8> %v10, ptr %ptr
  store volatile <16 x i8> %v9, ptr %ptr
  store volatile <16 x i8> %v8, ptr %ptr
  store volatile <16 x i8> %v7, ptr %ptr
  store volatile <16 x i8> %v6, ptr %ptr
  store volatile <16 x i8> %v5, ptr %ptr
  store volatile <16 x i8> %v4, ptr %ptr
  store volatile <16 x i8> %v3, ptr %ptr
  store volatile <16 x i8> %v2, ptr %ptr
  store volatile <16 x i8> %v1, ptr %ptr
  store volatile <16 x i8> %v0, ptr %ptr
  ret void
}

; Like f1, but no 16-byte slot should be needed, and no outgoing reg save
; area of 160 bytes.
define void @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: aghi %r15, -64
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
; CHECK: aghi %r15, 64
; CHECK: br %r14
  %v0 = load volatile <16 x i8>, ptr %ptr
  %v1 = load volatile <16 x i8>, ptr %ptr
  %v2 = load volatile <16 x i8>, ptr %ptr
  %v3 = load volatile <16 x i8>, ptr %ptr
  %v4 = load volatile <16 x i8>, ptr %ptr
  %v5 = load volatile <16 x i8>, ptr %ptr
  %v6 = load volatile <16 x i8>, ptr %ptr
  %v7 = load volatile <16 x i8>, ptr %ptr
  %v8 = load volatile <16 x i8>, ptr %ptr
  %v9 = load volatile <16 x i8>, ptr %ptr
  %v10 = load volatile <16 x i8>, ptr %ptr
  %v11 = load volatile <16 x i8>, ptr %ptr
  %v12 = load volatile <16 x i8>, ptr %ptr
  %v13 = load volatile <16 x i8>, ptr %ptr
  %v14 = load volatile <16 x i8>, ptr %ptr
  %v15 = load volatile <16 x i8>, ptr %ptr
  %v16 = load volatile <16 x i8>, ptr %ptr
  %v17 = load volatile <16 x i8>, ptr %ptr
  %v18 = load volatile <16 x i8>, ptr %ptr
  %v19 = load volatile <16 x i8>, ptr %ptr
  %v20 = load volatile <16 x i8>, ptr %ptr
  %v21 = load volatile <16 x i8>, ptr %ptr
  %v22 = load volatile <16 x i8>, ptr %ptr
  %v23 = load volatile <16 x i8>, ptr %ptr
  %v24 = load volatile <16 x i8>, ptr %ptr
  %v25 = load volatile <16 x i8>, ptr %ptr
  %v26 = load volatile <16 x i8>, ptr %ptr
  %v27 = load volatile <16 x i8>, ptr %ptr
  %v28 = load volatile <16 x i8>, ptr %ptr
  %v29 = load volatile <16 x i8>, ptr %ptr
  %v30 = load volatile <16 x i8>, ptr %ptr
  %v31 = load volatile <16 x i8>, ptr %ptr
  store volatile <16 x i8> %v31, ptr %ptr
  store volatile <16 x i8> %v30, ptr %ptr
  store volatile <16 x i8> %v29, ptr %ptr
  store volatile <16 x i8> %v28, ptr %ptr
  store volatile <16 x i8> %v27, ptr %ptr
  store volatile <16 x i8> %v26, ptr %ptr
  store volatile <16 x i8> %v25, ptr %ptr
  store volatile <16 x i8> %v24, ptr %ptr
  store volatile <16 x i8> %v23, ptr %ptr
  store volatile <16 x i8> %v22, ptr %ptr
  store volatile <16 x i8> %v21, ptr %ptr
  store volatile <16 x i8> %v20, ptr %ptr
  store volatile <16 x i8> %v19, ptr %ptr
  store volatile <16 x i8> %v18, ptr %ptr
  store volatile <16 x i8> %v17, ptr %ptr
  store volatile <16 x i8> %v16, ptr %ptr
  store volatile <16 x i8> %v15, ptr %ptr
  store volatile <16 x i8> %v14, ptr %ptr
  store volatile <16 x i8> %v13, ptr %ptr
  store volatile <16 x i8> %v12, ptr %ptr
  store volatile <16 x i8> %v11, ptr %ptr
  store volatile <16 x i8> %v10, ptr %ptr
  store volatile <16 x i8> %v9, ptr %ptr
  store volatile <16 x i8> %v8, ptr %ptr
  store volatile <16 x i8> %v7, ptr %ptr
  store volatile <16 x i8> %v6, ptr %ptr
  store volatile <16 x i8> %v5, ptr %ptr
  store volatile <16 x i8> %v4, ptr %ptr
  store volatile <16 x i8> %v3, ptr %ptr
  store volatile <16 x i8> %v2, ptr %ptr
  store volatile <16 x i8> %v1, ptr %ptr
  store volatile <16 x i8> %v0, ptr %ptr
  ret void
}

; Like f2, but only %f8 should be saved.
define void @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -8
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
; CHECK: aghi %r15, 8
; CHECK: br %r14
  %v0 = load volatile <16 x i8>, ptr %ptr
  %v1 = load volatile <16 x i8>, ptr %ptr
  %v2 = load volatile <16 x i8>, ptr %ptr
  %v3 = load volatile <16 x i8>, ptr %ptr
  %v4 = load volatile <16 x i8>, ptr %ptr
  %v5 = load volatile <16 x i8>, ptr %ptr
  %v6 = load volatile <16 x i8>, ptr %ptr
  %v7 = load volatile <16 x i8>, ptr %ptr
  %v8 = load volatile <16 x i8>, ptr %ptr
  %v16 = load volatile <16 x i8>, ptr %ptr
  %v17 = load volatile <16 x i8>, ptr %ptr
  %v18 = load volatile <16 x i8>, ptr %ptr
  %v19 = load volatile <16 x i8>, ptr %ptr
  %v20 = load volatile <16 x i8>, ptr %ptr
  %v21 = load volatile <16 x i8>, ptr %ptr
  %v22 = load volatile <16 x i8>, ptr %ptr
  %v23 = load volatile <16 x i8>, ptr %ptr
  %v24 = load volatile <16 x i8>, ptr %ptr
  %v25 = load volatile <16 x i8>, ptr %ptr
  %v26 = load volatile <16 x i8>, ptr %ptr
  %v27 = load volatile <16 x i8>, ptr %ptr
  %v28 = load volatile <16 x i8>, ptr %ptr
  %v29 = load volatile <16 x i8>, ptr %ptr
  %v30 = load volatile <16 x i8>, ptr %ptr
  %v31 = load volatile <16 x i8>, ptr %ptr
  store volatile <16 x i8> %v31, ptr %ptr
  store volatile <16 x i8> %v30, ptr %ptr
  store volatile <16 x i8> %v29, ptr %ptr
  store volatile <16 x i8> %v28, ptr %ptr
  store volatile <16 x i8> %v27, ptr %ptr
  store volatile <16 x i8> %v26, ptr %ptr
  store volatile <16 x i8> %v25, ptr %ptr
  store volatile <16 x i8> %v24, ptr %ptr
  store volatile <16 x i8> %v23, ptr %ptr
  store volatile <16 x i8> %v22, ptr %ptr
  store volatile <16 x i8> %v21, ptr %ptr
  store volatile <16 x i8> %v20, ptr %ptr
  store volatile <16 x i8> %v19, ptr %ptr
  store volatile <16 x i8> %v18, ptr %ptr
  store volatile <16 x i8> %v17, ptr %ptr
  store volatile <16 x i8> %v16, ptr %ptr
  store volatile <16 x i8> %v8, ptr %ptr
  store volatile <16 x i8> %v7, ptr %ptr
  store volatile <16 x i8> %v6, ptr %ptr
  store volatile <16 x i8> %v5, ptr %ptr
  store volatile <16 x i8> %v4, ptr %ptr
  store volatile <16 x i8> %v3, ptr %ptr
  store volatile <16 x i8> %v2, ptr %ptr
  store volatile <16 x i8> %v1, ptr %ptr
  store volatile <16 x i8> %v0, ptr %ptr
  ret void
}

; Like f2, but no registers should be saved.
define void @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r15
; CHECK: br %r14
  %v0 = load volatile <16 x i8>, ptr %ptr
  %v1 = load volatile <16 x i8>, ptr %ptr
  %v2 = load volatile <16 x i8>, ptr %ptr
  %v3 = load volatile <16 x i8>, ptr %ptr
  %v4 = load volatile <16 x i8>, ptr %ptr
  %v5 = load volatile <16 x i8>, ptr %ptr
  %v6 = load volatile <16 x i8>, ptr %ptr
  %v7 = load volatile <16 x i8>, ptr %ptr
  %v16 = load volatile <16 x i8>, ptr %ptr
  %v17 = load volatile <16 x i8>, ptr %ptr
  %v18 = load volatile <16 x i8>, ptr %ptr
  %v19 = load volatile <16 x i8>, ptr %ptr
  %v20 = load volatile <16 x i8>, ptr %ptr
  %v21 = load volatile <16 x i8>, ptr %ptr
  %v22 = load volatile <16 x i8>, ptr %ptr
  %v23 = load volatile <16 x i8>, ptr %ptr
  %v24 = load volatile <16 x i8>, ptr %ptr
  %v25 = load volatile <16 x i8>, ptr %ptr
  %v26 = load volatile <16 x i8>, ptr %ptr
  %v27 = load volatile <16 x i8>, ptr %ptr
  %v28 = load volatile <16 x i8>, ptr %ptr
  %v29 = load volatile <16 x i8>, ptr %ptr
  %v30 = load volatile <16 x i8>, ptr %ptr
  %v31 = load volatile <16 x i8>, ptr %ptr
  store volatile <16 x i8> %v31, ptr %ptr
  store volatile <16 x i8> %v30, ptr %ptr
  store volatile <16 x i8> %v29, ptr %ptr
  store volatile <16 x i8> %v28, ptr %ptr
  store volatile <16 x i8> %v27, ptr %ptr
  store volatile <16 x i8> %v26, ptr %ptr
  store volatile <16 x i8> %v25, ptr %ptr
  store volatile <16 x i8> %v24, ptr %ptr
  store volatile <16 x i8> %v23, ptr %ptr
  store volatile <16 x i8> %v22, ptr %ptr
  store volatile <16 x i8> %v21, ptr %ptr
  store volatile <16 x i8> %v20, ptr %ptr
  store volatile <16 x i8> %v19, ptr %ptr
  store volatile <16 x i8> %v18, ptr %ptr
  store volatile <16 x i8> %v17, ptr %ptr
  store volatile <16 x i8> %v16, ptr %ptr
  store volatile <16 x i8> %v7, ptr %ptr
  store volatile <16 x i8> %v6, ptr %ptr
  store volatile <16 x i8> %v5, ptr %ptr
  store volatile <16 x i8> %v4, ptr %ptr
  store volatile <16 x i8> %v3, ptr %ptr
  store volatile <16 x i8> %v2, ptr %ptr
  store volatile <16 x i8> %v1, ptr %ptr
  store volatile <16 x i8> %v0, ptr %ptr
  ret void
}
