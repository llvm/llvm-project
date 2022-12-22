; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s
; arm64 has a separate copy due to intrinsics

define <4 x i32> @copyTuple.QPair(ptr %a, ptr %b) {
; CHECK-LABEL: copyTuple.QPair:
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: ld2 { {{v[0-9]+}}.s, {{v[0-9]+}}.s }[{{[0-9]+}}], [x{{[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32.p0(<4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> <i32 2, i32 2, i32 2, i32 2>, i64 1, ptr %a)
  %extract = extractvalue { <4 x i32>, <4 x i32> } %vld, 0
  %vld1 = tail call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32.p0(<4 x i32> %extract, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i64 1, ptr %b)
  %vld1.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld1, 0
  ret <4 x i32> %vld1.fca.0.extract
}

define <4 x i32> @copyTuple.QTriple(ptr %a, ptr %b, <4 x i32> %c) {
; CHECK-LABEL: copyTuple.QTriple:
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: ld3 { {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s }[{{[0-9]+}}], [x{{[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32.p0(<4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, <4 x i32> %c, i64 1, ptr %a)
  %extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld, 0
  %vld1 = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32.p0(<4 x i32> %extract, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, i64 1, ptr %b)
  %vld1.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld1, 0
  ret <4 x i32> %vld1.fca.0.extract
}

define <4 x i32> @copyTuple.QQuad(ptr %a, ptr %b, <4 x i32> %c) {
; CHECK-LABEL: copyTuple.QQuad:
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: mov v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: ld4 { {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s }[{{[0-9]+}}], [x{{[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32.p0(<4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, <4 x i32> %c, <4 x i32> %c, i64 1, ptr %a)
  %extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld, 0
  %vld1 = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32.p0(<4 x i32> %extract, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, <4 x i32> %c, i64 1, ptr %b)
  %vld1.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld1, 0
  ret <4 x i32> %vld1.fca.0.extract
}

declare { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32.p0(<4 x i32>, <4 x i32>, i64, ptr)
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, i64, ptr)
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i64, ptr)
