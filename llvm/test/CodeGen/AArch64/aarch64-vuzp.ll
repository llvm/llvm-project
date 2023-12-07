; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8>, <16 x i8>)

; CHECK-LABEL: fun1:
; CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
define i32 @fun1() {
entry:
  %vtbl1.i.1 = tail call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> <i8 0, i8 16, i8 19, i8 4, i8 -65, i8 -65, i8 -71, i8 -71, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8> undef)
  %vuzp.i212.1 = shufflevector <16 x i8> %vtbl1.i.1, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %scevgep = getelementptr <8 x i8>, ptr undef, i64 1
  store <8 x i8> %vuzp.i212.1, ptr %scevgep, align 1
  ret i32 undef
}

; CHECK-LABEL: fun2:
; CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
define i32 @fun2() {
entry:
  %vtbl1.i.1 = tail call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> <i8 0, i8 16, i8 19, i8 4, i8 -65, i8 -65, i8 -71, i8 -71, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8> undef)
  %vuzp.i212.1 = shufflevector <16 x i8> %vtbl1.i.1, <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %scevgep = getelementptr <8 x i8>, ptr undef, i64 1
  store <8 x i8> %vuzp.i212.1, ptr %scevgep, align 1
  ret i32 undef
}

; CHECK-LABEL: fun3:
; CHECK-NOT: uzp1
define i32 @fun3() {
entry:
  %vtbl1.i.1 = tail call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> <i8 0, i8 16, i8 19, i8 4, i8 -65, i8 -65, i8 -71, i8 -71, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8> undef)
  %vuzp.i212.1 = shufflevector <16 x i8> %vtbl1.i.1, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 15>
  %scevgep = getelementptr <8 x i8>, ptr undef, i64 1
  store <8 x i8> %vuzp.i212.1, ptr %scevgep, align 1
  ret i32 undef
}

; CHECK-LABEL: fun4:
; CHECK-NOT: uzp2
define i32 @fun4() {
entry:
  %vtbl1.i.1 = tail call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> <i8 0, i8 16, i8 19, i8 4, i8 -65, i8 -65, i8 -71, i8 -71, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8> undef)
  %vuzp.i212.1 = shufflevector <16 x i8> %vtbl1.i.1, <16 x i8> undef, <8 x i32> <i32 3, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %scevgep = getelementptr <8 x i8>, ptr undef, i64 1
  store <8 x i8> %vuzp.i212.1, ptr %scevgep, align 1
  ret i32 undef
}

; CHECK-LABEL: pr36582:
; Check that this does not ICE.
define void @pr36582(ptr %p1, ptr %p2) {
entry:
  %wide.vec = load <8 x i8>, ptr %p1, align 1
  %strided.vec = shufflevector <8 x i8> %wide.vec, <8 x i8> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %y = zext <4 x i8> %strided.vec to <4 x i32>
  store <4 x i32> %y, ptr %p2, align 4
  ret void
}

; Check that this pattern is recognized as a VZIP and
; that the vector blend transform does not scramble the pattern.
; CHECK-LABEL: vzipNoBlend:
; CHECK: zip1
define <8 x i8> @vzipNoBlend(ptr %A, ptr %B) nounwind {
  %t = load <8 x i8>, ptr %A
  %vzip = shufflevector <8 x i8> %t, <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef, i8 undef>, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i8> %vzip
}

; CHECK-LABEL: vzipNoBlendCommutted:
; CHECK: zip1
define <8 x i8> @vzipNoBlendCommutted(ptr %A, ptr %B) nounwind {
  %t = load <8 x i8>, ptr %A
  %vzip = shufflevector <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef, i8 undef>, <8 x i8> %t, <8 x i32> <i32 8, i32 0, i32 9, i32 1, i32 10, i32 2, i32 11, i32 3>
  ret <8 x i8> %vzip
}

; CHECK-LABEL: vzipStillZExt:
; CHECK: zip1
define <8 x i8> @vzipStillZExt(ptr %A, ptr %B) nounwind {
  %t = load <8 x i8>, ptr %A
  %vzip = shufflevector <8 x i8> %t, <8 x i8> <i8 undef, i8 0, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef>, <8 x i32> <i32 0, i32 9, i32 1, i32 9, i32 2, i32 9, i32 3, i32 9>
  ret <8 x i8> %vzip
}

; CHECK-LABEL: vzipStillZExtCommutted:
; CHECK: zip1
define <8 x i8> @vzipStillZExtCommutted(ptr %A, ptr %B) nounwind {
  %t = load <8 x i8>, ptr %A
  %vzip = shufflevector <8 x i8> <i8 undef, i8 0, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef>, <8 x i8> %t, <8 x i32> <i32 8, i32 1, i32 9, i32 1, i32 10, i32 1, i32 11, i32 1>
  ret <8 x i8> %vzip
}
