; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Make sure all regs are spilled
define anyregcc void @anyregcc1() {
entry:
;CHECK-LABEL: anyregcc1
;CHECK: stmg %r2, %r15, 16(%r15)
;CHECK: vst %v0,
;CHECK: vst %v1,
;CHECK: vst %v2,
;CHECK: vst %v3,
;CHECK: vst %v4,
;CHECK: vst %v5,
;CHECK: vst %v6,
;CHECK: vst %v7,
;CHECK: vst %v8,
;CHECK: vst %v9,
;CHECK: vst %v10,
;CHECK: vst %v11,
;CHECK: vst %v12,
;CHECK: vst %v13,
;CHECK: vst %v14,
;CHECK: vst %v15,
;CHECK: vst %v16,
;CHECK: vst %v17,
;CHECK: vst %v18,
;CHECK: vst %v19,
;CHECK: vst %v20,
;CHECK: vst %v21,
;CHECK: vst %v22,
;CHECK: vst %v23,
;CHECK: vst %v24,
;CHECK: vst %v25,
;CHECK: vst %v26,
;CHECK: vst %v27,
;CHECK: vst %v28,
;CHECK: vst %v29,
;CHECK: vst %v30,
;CHECK: vst %v31,
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() nounwind
  ret void
}

; Make sure we don't spill any FPs
declare anyregcc void @foo()
define void @anyregcc2() {
entry:
;CHECK-LABEL: anyregcc2
;CHECK-NOT: std
;CHECK-NOT: vst
;CHECK: std %f8,
;CHECK-NEXT: std %f9,
;CHECK-NEXT: std %f10,
;CHECK-NEXT: std %f11,
;CHECK-NEXT: std %f12,
;CHECK-NEXT: std %f13,
;CHECK-NEXT: std %f14,
;CHECK-NEXT: std %f15,
;CHECK-NOT: std
;CHECK-NOT: vst
  %a0 = call <2 x i64> asm sideeffect "", "={v0}"() nounwind
  %a1 = call <2 x i64> asm sideeffect "", "={v1}"() nounwind
  %a2 = call <2 x i64> asm sideeffect "", "={v2}"() nounwind
  %a3 = call <2 x i64> asm sideeffect "", "={v3}"() nounwind
  %a4 = call <2 x i64> asm sideeffect "", "={v4}"() nounwind
  %a5 = call <2 x i64> asm sideeffect "", "={v5}"() nounwind
  %a6 = call <2 x i64> asm sideeffect "", "={v6}"() nounwind
  %a7 = call <2 x i64> asm sideeffect "", "={v7}"() nounwind
  %a8 = call <2 x i64> asm sideeffect "", "={v8}"() nounwind
  %a9 = call <2 x i64> asm sideeffect "", "={v9}"() nounwind
  %a10 = call <2 x i64> asm sideeffect "", "={v10}"() nounwind
  %a11 = call <2 x i64> asm sideeffect "", "={v11}"() nounwind
  %a12 = call <2 x i64> asm sideeffect "", "={v12}"() nounwind
  %a13 = call <2 x i64> asm sideeffect "", "={v13}"() nounwind
  %a14 = call <2 x i64> asm sideeffect "", "={v14}"() nounwind
  %a15 = call <2 x i64> asm sideeffect "", "={v15}"() nounwind
  %a16 = call <2 x i64> asm sideeffect "", "={v16}"() nounwind
  %a17 = call <2 x i64> asm sideeffect "", "={v17}"() nounwind
  %a18 = call <2 x i64> asm sideeffect "", "={v18}"() nounwind
  %a19 = call <2 x i64> asm sideeffect "", "={v19}"() nounwind
  %a20 = call <2 x i64> asm sideeffect "", "={v20}"() nounwind
  %a21 = call <2 x i64> asm sideeffect "", "={v21}"() nounwind
  %a22 = call <2 x i64> asm sideeffect "", "={v22}"() nounwind
  %a23 = call <2 x i64> asm sideeffect "", "={v23}"() nounwind
  %a24 = call <2 x i64> asm sideeffect "", "={v24}"() nounwind
  %a25 = call <2 x i64> asm sideeffect "", "={v25}"() nounwind
  %a26 = call <2 x i64> asm sideeffect "", "={v26}"() nounwind
  %a27 = call <2 x i64> asm sideeffect "", "={v27}"() nounwind
  %a28 = call <2 x i64> asm sideeffect "", "={v28}"() nounwind
  %a29 = call <2 x i64> asm sideeffect "", "={v29}"() nounwind
  %a30 = call <2 x i64> asm sideeffect "", "={v30}"() nounwind
  %a31 = call <2 x i64> asm sideeffect "", "={v31}"() nounwind
  call anyregcc void @foo()
  call void asm sideeffect "", "{v0},{v1},{v2},{v3},{v4},{v5},{v6},{v7},{v8},{v9},{v10},{v11},{v12},{v13},{v14},{v15},{v16},{v17},{v18},{v19},{v20},{v21},{v22},{v23},{v24},{v25},{v26},{v27},{v28},{v29},{v30},{v31}"(<2 x i64> %a0, <2 x i64> %a1, <2 x i64> %a2, <2 x i64> %a3, <2 x i64> %a4, <2 x i64> %a5, <2 x i64> %a6, <2 x i64> %a7, <2 x i64> %a8, <2 x i64> %a9, <2 x i64> %a10, <2 x i64> %a11, <2 x i64> %a12, <2 x i64> %a13, <2 x i64> %a14, <2 x i64> %a15, <2 x i64> %a16, <2 x i64> %a17, <2 x i64> %a18, <2 x i64> %a19, <2 x i64> %a20, <2 x i64> %a21, <2 x i64> %a22, <2 x i64> %a23, <2 x i64> %a24, <2 x i64> %a25, <2 x i64> %a26, <2 x i64> %a27, <2 x i64> %a28, <2 x i64> %a29, <2 x i64> %a30, <2 x i64> %a31)
  ret void
}

declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
