; RUN: llc -mtriple=arm64-apple-ios -o - %s | FileCheck %s

; The main purpose of this test is to find out whether copyPhysReg can deal with
; the memmove-like situation arising in tuples, where an early copy can clobber
; the value needed by a later one if the tuples overlap.

; We use dummy inline asm to force LLVM to generate a COPY between the registers
; we want by clobbering all the others.

define void @test_D1D2_from_D0D1(ptr %addr) #0 {
; CHECK-LABEL: test_D1D2_from_D0D1:
; CHECK: mov.8b v2, v1
; CHECK: mov.8b v1, v0
entry:
  %vec = tail call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %addr)
  %vec0 = extractvalue { <8 x i8>, <8 x i8> } %vec, 0
  %vec1 = extractvalue { <8 x i8>, <8 x i8> } %vec, 1
  tail call void asm sideeffect "", "~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)

  tail call void asm sideeffect "", "~{v0},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)
  ret void
}

define void @test_D0D1_from_D1D2(ptr %addr) #0 {
; CHECK-LABEL: test_D0D1_from_D1D2:
; CHECK: mov.8b v0, v1
; CHECK: mov.8b v1, v2
entry:
  %vec = tail call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %addr)
  %vec0 = extractvalue { <8 x i8>, <8 x i8> } %vec, 0
  %vec1 = extractvalue { <8 x i8>, <8 x i8> } %vec, 1
  tail call void asm sideeffect "", "~{v0},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)

  tail call void asm sideeffect "", "~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)
  ret void
}

define void @test_D0D1_from_D31D0(ptr %addr) #0 {
; CHECK-LABEL: test_D0D1_from_D31D0:
; CHECK: mov.8b v1, v0
; CHECK: mov.8b v0, v31
entry:
  %vec = tail call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %addr)
  %vec0 = extractvalue { <8 x i8>, <8 x i8> } %vec, 0
  %vec1 = extractvalue { <8 x i8>, <8 x i8> } %vec, 1
  tail call void asm sideeffect "", "~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)

  tail call void asm sideeffect "", "~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)
  ret void
}

define void @test_D31D0_from_D0D1(ptr %addr) #0 {
; CHECK-LABEL: test_D31D0_from_D0D1:
; CHECK: mov.8b v31, v0
; CHECK: mov.8b v0, v1
entry:
  %vec = tail call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %addr)
  %vec0 = extractvalue { <8 x i8>, <8 x i8> } %vec, 0
  %vec1 = extractvalue { <8 x i8>, <8 x i8> } %vec, 1
  tail call void asm sideeffect "", "~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)

  tail call void asm sideeffect "", "~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30}"()
  tail call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, ptr %addr)
  ret void
}

define void @test_D2D3D4_from_D0D1D2(ptr %addr) #0 {
; CHECK-LABEL: test_D2D3D4_from_D0D1D2:
; CHECK: mov.8b v4, v2
; CHECK: mov.8b v3, v1
; CHECK: mov.8b v2, v0
entry:
  %vec = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr %addr)
  %vec0 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vec, 0
  %vec1 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vec, 1
  %vec2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vec, 2

  tail call void asm sideeffect "", "~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, <8 x i8> %vec2, ptr %addr)

  tail call void asm sideeffect "", "~{v0},~{v1},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> %vec0, <8 x i8> %vec1, <8 x i8> %vec2, ptr %addr)
  ret void
}

define void @test_Q0Q1Q2_from_Q1Q2Q3(ptr %addr) #0 {
; CHECK-LABEL: test_Q0Q1Q2_from_Q1Q2Q3:
; CHECK: mov.16b v0, v1
; CHECK: mov.16b v1, v2
; CHECK: mov.16b v2, v3
entry:
  %vec = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr %addr)
  %vec0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vec, 0
  %vec1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vec, 1
  %vec2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vec, 2
  tail call void asm sideeffect "", "~{v0},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> %vec0, <16 x i8> %vec1, <16 x i8> %vec2, ptr %addr)

  tail call void asm sideeffect "", "~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> %vec0, <16 x i8> %vec1, <16 x i8> %vec2, ptr %addr)
  ret void
}

define void @test_Q1Q2Q3Q4_from_Q30Q31Q0Q1(ptr %addr) #0 {
; CHECK-LABEL: test_Q1Q2Q3Q4_from_Q30Q31Q0Q1:
; CHECK: mov.16b v4, v1
; CHECK: mov.16b v3, v0
; CHECK: mov.16b v2, v31
; CHECK: mov.16b v1, v30
  %vec = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr %addr)
  %vec0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vec, 0
  %vec1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vec, 1
  %vec2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vec, 2
  %vec3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vec, 3

  tail call void asm sideeffect "", "~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}"()
  tail call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> %vec0, <16 x i8> %vec1, <16 x i8> %vec2, <16 x i8> %vec3, ptr %addr)

  tail call void asm sideeffect "", "~{v0},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  tail call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> %vec0, <16 x i8> %vec1, <16 x i8> %vec2, <16 x i8> %vec3, ptr %addr)
  ret void
}

declare { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr)
declare { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr)
declare { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr)

declare void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8>, <8 x i8>, ptr)
declare void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8>, <8 x i8>, <8 x i8>, ptr)
declare void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, ptr)
declare void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, ptr)
