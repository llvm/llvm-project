; RUN: opt -passes=debugify %s -o - | llc -mtriple=aarch64 -global-isel -stop-after=aarch64-prelegalizer-combiner -o - | FileCheck %s

; Check that DBG_VALUEs using a dead G_BITCAST or G_BUILD_VECTOR are salvaged
; into per-element fragments instead of being set to undef.

; CHECK-LABEL: name: bitcast_build_vector
; CHECK: [[A:%[0-9]+]]:_(i64) = COPY $x0
; CHECK-NEXT: [[B:%[0-9]+]]:_(i64) = COPY $x1
; CHECK-NEXT: DBG_VALUE [[A]](i64), $noreg, [[V0:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, [[V0]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
; CHECK-NEXT: DBG_VALUE [[A]](i64), $noreg, [[V1:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; CHECK-NEXT: DBG_VALUE [[B]](i64), $noreg, [[V1]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
; CHECK-NEXT: DBG_VALUE [[A]](i64), $noreg, [[X:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; CHECK-NEXT: DBG_VALUE [[B]](i64), $noreg, [[X]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
define i128 @bitcast_build_vector(i64 %a, i64 %b) {
  %v0 = insertelement <2 x i64> poison, i64 %a, i64 0
  %v1 = insertelement <2 x i64> %v0, i64 %b, i64 1
  %x = bitcast <2 x i64> %v1 to i128
  ret i128 %x
}

; CHECK-LABEL: name: build_vector
; CHECK: [[A:%[0-9]+]]:_(i32) = COPY $w0
; CHECK: [[D:%[0-9]+]]:_(i32) = COPY $w3
; CHECK-NEXT: DBG_VALUE [[A]](i32), $noreg, [[V0:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, [[V0]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, [[V0]], !DIExpression(DW_OP_LLVM_fragment, 64, 32)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, [[V0]], !DIExpression(DW_OP_LLVM_fragment, 96, 32)
; CHECK-NEXT: DBG_VALUE [[A]](i32), $noreg, [[V1:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE $w1, $noreg, [[V1]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, [[V1]], !DIExpression(DW_OP_LLVM_fragment, 64, 32)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, [[V1]], !DIExpression(DW_OP_LLVM_fragment, 96, 32)
; CHECK-NEXT: DBG_VALUE [[A]](i32), $noreg, [[V2:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE $w1, $noreg, [[V2]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: DBG_VALUE $w2, $noreg, [[V2]], !DIExpression(DW_OP_LLVM_fragment, 64, 32)
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, [[V2]], !DIExpression(DW_OP_LLVM_fragment, 96, 32)
; CHECK-NEXT: DBG_VALUE [[A]](i32), $noreg, [[V3:![0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT: DBG_VALUE $w1, $noreg, [[V3]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT: DBG_VALUE $w2, $noreg, [[V3]], !DIExpression(DW_OP_LLVM_fragment, 64, 32)
; CHECK-NEXT: DBG_VALUE [[D]](i32), $noreg, [[V3]], !DIExpression(DW_OP_LLVM_fragment, 96, 32)
define i32 @build_vector(i32 %a, i32 %b, i32 %c, i32 %d) {
  %v0 = insertelement <4 x i32> poison, i32 %a, i64 0
  %v1 = insertelement <4 x i32> %v0, i32 %b, i64 1
  %v2 = insertelement <4 x i32> %v1, i32 %c, i64 2
  %v3 = insertelement <4 x i32> %v2, i32 %d, i64 3
  %e0 = extractelement <4 x i32> %v3, i64 0
  %e3 = extractelement <4 x i32> %v3, i64 3
  %r = add i32 %e0, %e3
  ret i32 %r
}
