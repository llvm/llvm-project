; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvpermi.w(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvpermi_w_lo(<8 x i32> %va, <8 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvpermi.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvpermi.w(<8 x i32> %va, <8 x i32> %vb, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvpermi_w_hi(<8 x i32> %va, <8 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvpermi.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvpermi.w(<8 x i32> %va, <8 x i32> %vb, i32 256)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvpermi.d(<4 x i64>, i32)

define <4 x i64> @lasx_xvpermi_d_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpermi.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvpermi.d(<4 x i64> %va, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvpermi_d_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvpermi.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvpermi.d(<4 x i64> %va, i32 256)
  ret <4 x i64> %res
}

declare <32 x i8> @llvm.loongarch.lasx.xvpermi.q(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvpermi_q_lo(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvpermi.q: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvpermi.q(<32 x i8> %va, <32 x i8> %vb, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvpermi_q_hi(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvpermi.q: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvpermi.q(<32 x i8> %va, <32 x i8> %vb, i32 256)
  ret <32 x i8> %res
}
