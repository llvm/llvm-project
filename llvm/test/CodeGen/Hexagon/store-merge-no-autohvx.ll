; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOHVX --implicit-check-not=memw
; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b -hexagon-autohvx < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,AUTOHVX --implicit-check-not=memw

; Store merging may only form an HVX vector when HVX auto-vectorization is
; enabled for the function. The 32 i32 copies below fill exactly one 128-byte
; HVX vector. Without HVX, each 8-byte aligned pair still merges into an i64.

; CHECK-LABEL: copy32:
; NOHVX-NOT: vmem
; NOHVX-COUNT-16: memd(r0+#{{[0-9]+}}) = r{{[0-9]+}}:{{[0-9]+}}
; NOHVX-NOT: vmem
; AUTOHVX: vmemu
define void @copy32(ptr %dst, ptr %src) #0 {
  %ps0 = getelementptr inbounds i32, ptr %src, i32 0
  %v0 = load i32, ptr %ps0, align 8
  %ps1 = getelementptr inbounds i32, ptr %src, i32 1
  %v1 = load i32, ptr %ps1, align 4
  %ps2 = getelementptr inbounds i32, ptr %src, i32 2
  %v2 = load i32, ptr %ps2, align 8
  %ps3 = getelementptr inbounds i32, ptr %src, i32 3
  %v3 = load i32, ptr %ps3, align 4
  %ps4 = getelementptr inbounds i32, ptr %src, i32 4
  %v4 = load i32, ptr %ps4, align 8
  %ps5 = getelementptr inbounds i32, ptr %src, i32 5
  %v5 = load i32, ptr %ps5, align 4
  %ps6 = getelementptr inbounds i32, ptr %src, i32 6
  %v6 = load i32, ptr %ps6, align 8
  %ps7 = getelementptr inbounds i32, ptr %src, i32 7
  %v7 = load i32, ptr %ps7, align 4
  %ps8 = getelementptr inbounds i32, ptr %src, i32 8
  %v8 = load i32, ptr %ps8, align 8
  %ps9 = getelementptr inbounds i32, ptr %src, i32 9
  %v9 = load i32, ptr %ps9, align 4
  %ps10 = getelementptr inbounds i32, ptr %src, i32 10
  %v10 = load i32, ptr %ps10, align 8
  %ps11 = getelementptr inbounds i32, ptr %src, i32 11
  %v11 = load i32, ptr %ps11, align 4
  %ps12 = getelementptr inbounds i32, ptr %src, i32 12
  %v12 = load i32, ptr %ps12, align 8
  %ps13 = getelementptr inbounds i32, ptr %src, i32 13
  %v13 = load i32, ptr %ps13, align 4
  %ps14 = getelementptr inbounds i32, ptr %src, i32 14
  %v14 = load i32, ptr %ps14, align 8
  %ps15 = getelementptr inbounds i32, ptr %src, i32 15
  %v15 = load i32, ptr %ps15, align 4
  %ps16 = getelementptr inbounds i32, ptr %src, i32 16
  %v16 = load i32, ptr %ps16, align 8
  %ps17 = getelementptr inbounds i32, ptr %src, i32 17
  %v17 = load i32, ptr %ps17, align 4
  %ps18 = getelementptr inbounds i32, ptr %src, i32 18
  %v18 = load i32, ptr %ps18, align 8
  %ps19 = getelementptr inbounds i32, ptr %src, i32 19
  %v19 = load i32, ptr %ps19, align 4
  %ps20 = getelementptr inbounds i32, ptr %src, i32 20
  %v20 = load i32, ptr %ps20, align 8
  %ps21 = getelementptr inbounds i32, ptr %src, i32 21
  %v21 = load i32, ptr %ps21, align 4
  %ps22 = getelementptr inbounds i32, ptr %src, i32 22
  %v22 = load i32, ptr %ps22, align 8
  %ps23 = getelementptr inbounds i32, ptr %src, i32 23
  %v23 = load i32, ptr %ps23, align 4
  %ps24 = getelementptr inbounds i32, ptr %src, i32 24
  %v24 = load i32, ptr %ps24, align 8
  %ps25 = getelementptr inbounds i32, ptr %src, i32 25
  %v25 = load i32, ptr %ps25, align 4
  %ps26 = getelementptr inbounds i32, ptr %src, i32 26
  %v26 = load i32, ptr %ps26, align 8
  %ps27 = getelementptr inbounds i32, ptr %src, i32 27
  %v27 = load i32, ptr %ps27, align 4
  %ps28 = getelementptr inbounds i32, ptr %src, i32 28
  %v28 = load i32, ptr %ps28, align 8
  %ps29 = getelementptr inbounds i32, ptr %src, i32 29
  %v29 = load i32, ptr %ps29, align 4
  %ps30 = getelementptr inbounds i32, ptr %src, i32 30
  %v30 = load i32, ptr %ps30, align 8
  %ps31 = getelementptr inbounds i32, ptr %src, i32 31
  %v31 = load i32, ptr %ps31, align 4
  %pd0 = getelementptr inbounds i32, ptr %dst, i32 0
  store i32 %v0, ptr %pd0, align 8
  %pd1 = getelementptr inbounds i32, ptr %dst, i32 1
  store i32 %v1, ptr %pd1, align 4
  %pd2 = getelementptr inbounds i32, ptr %dst, i32 2
  store i32 %v2, ptr %pd2, align 8
  %pd3 = getelementptr inbounds i32, ptr %dst, i32 3
  store i32 %v3, ptr %pd3, align 4
  %pd4 = getelementptr inbounds i32, ptr %dst, i32 4
  store i32 %v4, ptr %pd4, align 8
  %pd5 = getelementptr inbounds i32, ptr %dst, i32 5
  store i32 %v5, ptr %pd5, align 4
  %pd6 = getelementptr inbounds i32, ptr %dst, i32 6
  store i32 %v6, ptr %pd6, align 8
  %pd7 = getelementptr inbounds i32, ptr %dst, i32 7
  store i32 %v7, ptr %pd7, align 4
  %pd8 = getelementptr inbounds i32, ptr %dst, i32 8
  store i32 %v8, ptr %pd8, align 8
  %pd9 = getelementptr inbounds i32, ptr %dst, i32 9
  store i32 %v9, ptr %pd9, align 4
  %pd10 = getelementptr inbounds i32, ptr %dst, i32 10
  store i32 %v10, ptr %pd10, align 8
  %pd11 = getelementptr inbounds i32, ptr %dst, i32 11
  store i32 %v11, ptr %pd11, align 4
  %pd12 = getelementptr inbounds i32, ptr %dst, i32 12
  store i32 %v12, ptr %pd12, align 8
  %pd13 = getelementptr inbounds i32, ptr %dst, i32 13
  store i32 %v13, ptr %pd13, align 4
  %pd14 = getelementptr inbounds i32, ptr %dst, i32 14
  store i32 %v14, ptr %pd14, align 8
  %pd15 = getelementptr inbounds i32, ptr %dst, i32 15
  store i32 %v15, ptr %pd15, align 4
  %pd16 = getelementptr inbounds i32, ptr %dst, i32 16
  store i32 %v16, ptr %pd16, align 8
  %pd17 = getelementptr inbounds i32, ptr %dst, i32 17
  store i32 %v17, ptr %pd17, align 4
  %pd18 = getelementptr inbounds i32, ptr %dst, i32 18
  store i32 %v18, ptr %pd18, align 8
  %pd19 = getelementptr inbounds i32, ptr %dst, i32 19
  store i32 %v19, ptr %pd19, align 4
  %pd20 = getelementptr inbounds i32, ptr %dst, i32 20
  store i32 %v20, ptr %pd20, align 8
  %pd21 = getelementptr inbounds i32, ptr %dst, i32 21
  store i32 %v21, ptr %pd21, align 4
  %pd22 = getelementptr inbounds i32, ptr %dst, i32 22
  store i32 %v22, ptr %pd22, align 8
  %pd23 = getelementptr inbounds i32, ptr %dst, i32 23
  store i32 %v23, ptr %pd23, align 4
  %pd24 = getelementptr inbounds i32, ptr %dst, i32 24
  store i32 %v24, ptr %pd24, align 8
  %pd25 = getelementptr inbounds i32, ptr %dst, i32 25
  store i32 %v25, ptr %pd25, align 4
  %pd26 = getelementptr inbounds i32, ptr %dst, i32 26
  store i32 %v26, ptr %pd26, align 8
  %pd27 = getelementptr inbounds i32, ptr %dst, i32 27
  store i32 %v27, ptr %pd27, align 4
  %pd28 = getelementptr inbounds i32, ptr %dst, i32 28
  store i32 %v28, ptr %pd28, align 8
  %pd29 = getelementptr inbounds i32, ptr %dst, i32 29
  store i32 %v29, ptr %pd29, align 4
  %pd30 = getelementptr inbounds i32, ptr %dst, i32 30
  store i32 %v30, ptr %pd30, align 8
  %pd31 = getelementptr inbounds i32, ptr %dst, i32 31
  store i32 %v31, ptr %pd31, align 4
  ret void
}

; CHECK-LABEL: copy32_hmx:
; CHECK-NOT: vmem
; CHECK-COUNT-16: memd(r0+#{{[0-9]+}}) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK-NOT: vmem
define void @copy32_hmx(ptr %dst, ptr %src) #1 {
  %ps0 = getelementptr inbounds i32, ptr %src, i32 0
  %v0 = load i32, ptr %ps0, align 8
  %ps1 = getelementptr inbounds i32, ptr %src, i32 1
  %v1 = load i32, ptr %ps1, align 4
  %ps2 = getelementptr inbounds i32, ptr %src, i32 2
  %v2 = load i32, ptr %ps2, align 8
  %ps3 = getelementptr inbounds i32, ptr %src, i32 3
  %v3 = load i32, ptr %ps3, align 4
  %ps4 = getelementptr inbounds i32, ptr %src, i32 4
  %v4 = load i32, ptr %ps4, align 8
  %ps5 = getelementptr inbounds i32, ptr %src, i32 5
  %v5 = load i32, ptr %ps5, align 4
  %ps6 = getelementptr inbounds i32, ptr %src, i32 6
  %v6 = load i32, ptr %ps6, align 8
  %ps7 = getelementptr inbounds i32, ptr %src, i32 7
  %v7 = load i32, ptr %ps7, align 4
  %ps8 = getelementptr inbounds i32, ptr %src, i32 8
  %v8 = load i32, ptr %ps8, align 8
  %ps9 = getelementptr inbounds i32, ptr %src, i32 9
  %v9 = load i32, ptr %ps9, align 4
  %ps10 = getelementptr inbounds i32, ptr %src, i32 10
  %v10 = load i32, ptr %ps10, align 8
  %ps11 = getelementptr inbounds i32, ptr %src, i32 11
  %v11 = load i32, ptr %ps11, align 4
  %ps12 = getelementptr inbounds i32, ptr %src, i32 12
  %v12 = load i32, ptr %ps12, align 8
  %ps13 = getelementptr inbounds i32, ptr %src, i32 13
  %v13 = load i32, ptr %ps13, align 4
  %ps14 = getelementptr inbounds i32, ptr %src, i32 14
  %v14 = load i32, ptr %ps14, align 8
  %ps15 = getelementptr inbounds i32, ptr %src, i32 15
  %v15 = load i32, ptr %ps15, align 4
  %ps16 = getelementptr inbounds i32, ptr %src, i32 16
  %v16 = load i32, ptr %ps16, align 8
  %ps17 = getelementptr inbounds i32, ptr %src, i32 17
  %v17 = load i32, ptr %ps17, align 4
  %ps18 = getelementptr inbounds i32, ptr %src, i32 18
  %v18 = load i32, ptr %ps18, align 8
  %ps19 = getelementptr inbounds i32, ptr %src, i32 19
  %v19 = load i32, ptr %ps19, align 4
  %ps20 = getelementptr inbounds i32, ptr %src, i32 20
  %v20 = load i32, ptr %ps20, align 8
  %ps21 = getelementptr inbounds i32, ptr %src, i32 21
  %v21 = load i32, ptr %ps21, align 4
  %ps22 = getelementptr inbounds i32, ptr %src, i32 22
  %v22 = load i32, ptr %ps22, align 8
  %ps23 = getelementptr inbounds i32, ptr %src, i32 23
  %v23 = load i32, ptr %ps23, align 4
  %ps24 = getelementptr inbounds i32, ptr %src, i32 24
  %v24 = load i32, ptr %ps24, align 8
  %ps25 = getelementptr inbounds i32, ptr %src, i32 25
  %v25 = load i32, ptr %ps25, align 4
  %ps26 = getelementptr inbounds i32, ptr %src, i32 26
  %v26 = load i32, ptr %ps26, align 8
  %ps27 = getelementptr inbounds i32, ptr %src, i32 27
  %v27 = load i32, ptr %ps27, align 4
  %ps28 = getelementptr inbounds i32, ptr %src, i32 28
  %v28 = load i32, ptr %ps28, align 8
  %ps29 = getelementptr inbounds i32, ptr %src, i32 29
  %v29 = load i32, ptr %ps29, align 4
  %ps30 = getelementptr inbounds i32, ptr %src, i32 30
  %v30 = load i32, ptr %ps30, align 8
  %ps31 = getelementptr inbounds i32, ptr %src, i32 31
  %v31 = load i32, ptr %ps31, align 4
  %pd0 = getelementptr inbounds i32, ptr %dst, i32 0
  store i32 %v0, ptr %pd0, align 8
  %pd1 = getelementptr inbounds i32, ptr %dst, i32 1
  store i32 %v1, ptr %pd1, align 4
  %pd2 = getelementptr inbounds i32, ptr %dst, i32 2
  store i32 %v2, ptr %pd2, align 8
  %pd3 = getelementptr inbounds i32, ptr %dst, i32 3
  store i32 %v3, ptr %pd3, align 4
  %pd4 = getelementptr inbounds i32, ptr %dst, i32 4
  store i32 %v4, ptr %pd4, align 8
  %pd5 = getelementptr inbounds i32, ptr %dst, i32 5
  store i32 %v5, ptr %pd5, align 4
  %pd6 = getelementptr inbounds i32, ptr %dst, i32 6
  store i32 %v6, ptr %pd6, align 8
  %pd7 = getelementptr inbounds i32, ptr %dst, i32 7
  store i32 %v7, ptr %pd7, align 4
  %pd8 = getelementptr inbounds i32, ptr %dst, i32 8
  store i32 %v8, ptr %pd8, align 8
  %pd9 = getelementptr inbounds i32, ptr %dst, i32 9
  store i32 %v9, ptr %pd9, align 4
  %pd10 = getelementptr inbounds i32, ptr %dst, i32 10
  store i32 %v10, ptr %pd10, align 8
  %pd11 = getelementptr inbounds i32, ptr %dst, i32 11
  store i32 %v11, ptr %pd11, align 4
  %pd12 = getelementptr inbounds i32, ptr %dst, i32 12
  store i32 %v12, ptr %pd12, align 8
  %pd13 = getelementptr inbounds i32, ptr %dst, i32 13
  store i32 %v13, ptr %pd13, align 4
  %pd14 = getelementptr inbounds i32, ptr %dst, i32 14
  store i32 %v14, ptr %pd14, align 8
  %pd15 = getelementptr inbounds i32, ptr %dst, i32 15
  store i32 %v15, ptr %pd15, align 4
  %pd16 = getelementptr inbounds i32, ptr %dst, i32 16
  store i32 %v16, ptr %pd16, align 8
  %pd17 = getelementptr inbounds i32, ptr %dst, i32 17
  store i32 %v17, ptr %pd17, align 4
  %pd18 = getelementptr inbounds i32, ptr %dst, i32 18
  store i32 %v18, ptr %pd18, align 8
  %pd19 = getelementptr inbounds i32, ptr %dst, i32 19
  store i32 %v19, ptr %pd19, align 4
  %pd20 = getelementptr inbounds i32, ptr %dst, i32 20
  store i32 %v20, ptr %pd20, align 8
  %pd21 = getelementptr inbounds i32, ptr %dst, i32 21
  store i32 %v21, ptr %pd21, align 4
  %pd22 = getelementptr inbounds i32, ptr %dst, i32 22
  store i32 %v22, ptr %pd22, align 8
  %pd23 = getelementptr inbounds i32, ptr %dst, i32 23
  store i32 %v23, ptr %pd23, align 4
  %pd24 = getelementptr inbounds i32, ptr %dst, i32 24
  store i32 %v24, ptr %pd24, align 8
  %pd25 = getelementptr inbounds i32, ptr %dst, i32 25
  store i32 %v25, ptr %pd25, align 4
  %pd26 = getelementptr inbounds i32, ptr %dst, i32 26
  store i32 %v26, ptr %pd26, align 8
  %pd27 = getelementptr inbounds i32, ptr %dst, i32 27
  store i32 %v27, ptr %pd27, align 4
  %pd28 = getelementptr inbounds i32, ptr %dst, i32 28
  store i32 %v28, ptr %pd28, align 8
  %pd29 = getelementptr inbounds i32, ptr %dst, i32 29
  store i32 %v29, ptr %pd29, align 4
  %pd30 = getelementptr inbounds i32, ptr %dst, i32 30
  store i32 %v30, ptr %pd30, align 8
  %pd31 = getelementptr inbounds i32, ptr %dst, i32 31
  store i32 %v31, ptr %pd31, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "hexagon_hmx" }
