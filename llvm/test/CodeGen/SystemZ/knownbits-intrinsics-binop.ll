; Test that DAGCombiner gets helped by computeKnownBitsForTargetNode() with
; vector intrinsics.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s

declare {<16 x i8>, i32} @llvm.s390.vpkshs(<8 x i16>, <8 x i16>)
declare {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32>, <4 x i32>)
declare {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64>, <2 x i64>)

; PACKS_CC (operand elements are 0): i64 -> i32
define <4 x i32> @f0() {
; CHECK-LABEL: f0:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64> <i64 0, i64 0>, <2 x i64> <i64 0, i64 0>)
  %extr = extractvalue {<4 x i32>, i32} %call, 0
  %and = and <4 x i32> %extr, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKS_CC (operand elements are 1): i64 -> i32
; NOTE: The vector AND is optimized away, but vrepig+vpksgs is used instead
; of vrepif. Similarly for more test cases below.
define <4 x i32> @f1() {
; CHECK-LABEL: f1:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepig %v0, 1
; CHECK-NEXT:  vpksgs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64> <i64 1, i64 1>, <2 x i64> <i64 1, i64 1>)
  %extr = extractvalue {<4 x i32>, i32} %call, 0
  %and = and <4 x i32> %extr, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKS_CC (operand elements are 0): i32 -> i16
define <8 x i16> @f2() {
; CHECK-LABEL: f2:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32> <i32 0, i32 0, i32 0, i32 0>,
                                                  <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  %extr = extractvalue {<8 x i16>, i32} %call, 0
  %and = and <8 x i16> %extr, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKS_CC (operand elements are 1): i32 -> i16
define <8 x i16> @f3() {
; CHECK-LABEL: f3:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepif %v0, 1
; CHECK-NEXT:  vpksfs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32> <i32 1, i32 1, i32 1, i32 1>,
                                                  <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  %extr = extractvalue {<8 x i16>, i32} %call, 0
  %and = and <8 x i16> %extr, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKS_CC (operand elements are 0): i16 -> i8
define <16 x i8> @f4() {
; CHECK-LABEL: f4:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpkshs(
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>,
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
  %extr = extractvalue {<16 x i8>, i32} %call, 0
  %and = and <16 x i8> %extr, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

; PACKS_CC (operand elements are 1): i16 -> i8
define <16 x i8> @f5() {
; CHECK-LABEL: f5:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepih %v0, 1
; CHECK-NEXT:  vpkshs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpkshs(
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>,
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  %extr = extractvalue {<16 x i8>, i32} %call, 0
  %and = and <16 x i8> %extr, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

declare {<16 x i8>, i32} @llvm.s390.vpklshs(<8 x i16>, <8 x i16>)
declare {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32>, <4 x i32>)
declare {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64>, <2 x i64>)

; PACKLS_CC (operand elements are 0): i64 -> i32
define <4 x i32> @f6() {
; CHECK-LABEL: f6:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64> <i64 0, i64 0>, <2 x i64> <i64 0, i64 0>)
  %extr = extractvalue {<4 x i32>, i32} %call, 0
  %and = and <4 x i32> %extr, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKLS_CC (operand elements are 1): i64 -> i32
define <4 x i32> @f7() {
; CHECK-LABEL: f7:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepig %v0, 1
; CHECK-NEXT:  vpklsgs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64> <i64 1, i64 1>, <2 x i64> <i64 1, i64 1>)
  %extr = extractvalue {<4 x i32>, i32} %call, 0
  %and = and <4 x i32> %extr, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKLS_CC (operand elements are 0): i32 -> i16
define <8 x i16> @f8() {
; CHECK-LABEL: f8:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32> <i32 0, i32 0, i32 0, i32 0>,
                                                  <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  %extr = extractvalue {<8 x i16>, i32} %call, 0
  %and = and <8 x i16> %extr, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKLS_CC (operand elements are 1): i32 -> i16
define <8 x i16> @f9() {
; CHECK-LABEL: f9:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepif %v0, 1
; CHECK-NEXT:  vpklsfs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32> <i32 1, i32 1, i32 1, i32 1>,
                                                  <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  %extr = extractvalue {<8 x i16>, i32} %call, 0
  %and = and <8 x i16> %extr, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKLS_CC (operand elements are 0): i16 -> i8
define <16 x i8> @f10() {
; CHECK-LABEL: f10:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpklshs(
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>,
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
  %extr = extractvalue {<16 x i8>, i32} %call, 0
  %and = and <16 x i8> %extr, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

; PACKLS_CC (operand elements are 1): i16 -> i8
define <16 x i8> @f11() {
; CHECK-LABEL: f11:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepih %v0, 1
; CHECK-NEXT:  vpklshs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpklshs(
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>,
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  %extr = extractvalue {<16 x i8>, i32} %call, 0
  %and = and <16 x i8> %extr, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

declare <16 x i8> @llvm.s390.vpksh(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.s390.vpksf(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.s390.vpksg(<2 x i64>, <2 x i64>)

; PACKS (operand elements are 0): i64 -> i32
define <4 x i32> @f12() {
; CHECK-LABEL: f12:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call <4 x i32> @llvm.s390.vpksg(<2 x i64> <i64 0, i64 0>, <2 x i64> <i64 0, i64 0>)
  %and = and <4 x i32> %call, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKS (operand elements are 1): i64 -> i32
define <4 x i32> @f13() {
; CHECK-LABEL: f13:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepig %v0, 1
; CHECK-NEXT:  vpksg %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <4 x i32> @llvm.s390.vpksg(<2 x i64> <i64 1, i64 1>, <2 x i64> <i64 1, i64 1>)
  %and = and <4 x i32> %call, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKS (operand elements are 0): i32 -> i16
define <8 x i16> @f14() {
; CHECK-LABEL: f14:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call <8 x i16> @llvm.s390.vpksf(<4 x i32> <i32 0, i32 0, i32 0, i32 0>,
                                          <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  %and = and <8 x i16> %call, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKS (operand elements are 1): i32 -> i16
define <8 x i16> @f15() {
; CHECK-LABEL: f15:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepif %v0, 1
; CHECK-NEXT:  vpksf %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <8 x i16> @llvm.s390.vpksf(<4 x i32> <i32 1, i32 1, i32 1, i32 1>,
                                          <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  %and = and <8 x i16> %call, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKS (operand elements are 0): i16 -> i8
define <16 x i8> @f16() {
; CHECK-LABEL: f16:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call <16 x i8> @llvm.s390.vpksh(
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>,
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
  %and = and <16 x i8> %call, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

; PACKS (operand elements are 1): i16 -> i8
define <16 x i8> @f17() {
; CHECK-LABEL: f17:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepih %v0, 1
; CHECK-NEXT:  vpksh %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <16 x i8> @llvm.s390.vpksh(
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>,
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  %and = and <16 x i8> %call, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

declare <16 x i8> @llvm.s390.vpklsh(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.s390.vpklsf(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.s390.vpklsg(<2 x i64>, <2 x i64>)

; PACKLS (operand elements are 0): i64 -> i32
define <4 x i32> @f18() {
; CHECK-LABEL: f18:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call <4 x i32> @llvm.s390.vpklsg(<2 x i64> <i64 0, i64 0>, <2 x i64> <i64 0, i64 0>)
  %and = and <4 x i32> %call, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKLS (operand elements are 1): i64 -> i32
define <4 x i32> @f19() {
; CHECK-LABEL: f19:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepig %v0, 1
; CHECK-NEXT:  vpklsg %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <4 x i32> @llvm.s390.vpklsg(<2 x i64> <i64 1, i64 1>, <2 x i64> <i64 1, i64 1>)
  %and = and <4 x i32> %call, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; PACKLS (operand elements are 0): i32 -> i16
define <8 x i16> @f20() {
; CHECK-LABEL: f20:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call <8 x i16> @llvm.s390.vpklsf(<4 x i32> <i32 0, i32 0, i32 0, i32 0>,
                                           <4 x i32> <i32 0, i32 0, i32 0, i32 0>)
  %and = and <8 x i16> %call, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKLS (operand elements are 1): i32 -> i16
define <8 x i16> @f21() {
; CHECK-LABEL: f21:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepif %v0, 1
; CHECK-NEXT:  vpklsf %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <8 x i16> @llvm.s390.vpklsf(<4 x i32> <i32 1, i32 1, i32 1, i32 1>,
                                           <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  %and = and <8 x i16> %call, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; PACKLS (operand elements are 0): i16 -> i8
define <16 x i8> @f22() {
; CHECK-LABEL: f22:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %call = call <16 x i8> @llvm.s390.vpklsh(
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>,
                <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
  %and = and <16 x i8> %call, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

; PACKLS (operand elements are 1): i16 -> i8
define <16 x i8> @f23() {
; CHECK-LABEL: f23:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vrepih %v0, 1
; CHECK-NEXT:  vpklsh %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <16 x i8> @llvm.s390.vpklsh(
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>,
                <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  %and = and <16 x i8> %call, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %and
}

declare <2 x i64> @llvm.s390.vpdi(<2 x i64>, <2 x i64>, i32)

; VPDI (operand elements are 0):
define <2 x i64> @f24() {
; CHECK-LABEL: f24:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: vgbm %v24, 0
; CHECK-NEXT: br %r14
  %perm = call <2 x i64> @llvm.s390.vpdi(<2 x i64> <i64 0, i64 0>,
                                         <2 x i64> <i64 0, i64 0>, i32 0)
  %res = and <2 x i64> %perm, <i64 1, i64 1>
  ret <2 x i64> %res
}

; VPDI (operand elements are 1):
define <2 x i64> @f25() {
; CHECK-LABEL: f25:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: vrepig %v0, 1
; CHECK-NEXT: vpdi %v24, %v0, %v0, 0
; CHECK-NEXT: br %r14
  %perm = call <2 x i64> @llvm.s390.vpdi(<2 x i64> <i64 1, i64 1>,
                                         <2 x i64> <i64 1, i64 1>, i32 0)
  %res = and <2 x i64> %perm, <i64 1, i64 1>
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.s390.vsldb(<16 x i8>, <16 x i8>, i32)

; VSLDB (operand elements are 0):
define <16 x i8> @f26() {
; CHECK-LABEL: f26:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: vgbm %v24, 0
; CHECK-NEXT: br %r14
  %shfd = call <16 x i8> @llvm.s390.vsldb(<16 x i8>
                 <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                  i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8>
                 <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                  i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>,
                  i32 1)

  %res = and <16 x i8> %shfd, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %res
}

; VSLDB (operand elements are 1):
define <16 x i8> @f27() {
; CHECK-LABEL: f27:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: vrepib %v0, 1
; CHECK-NEXT: vsldb %v24, %v0, %v0, 1
; CHECK-NEXT: br %r14
  %shfd = call <16 x i8> @llvm.s390.vsldb(<16 x i8>
                 <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                  i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8>
                 <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                  i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>,
                  i32 1)

  %res = and <16 x i8> %shfd, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %res
}

; Test that intrinsic CC result is recognized.
define i32 @f28(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: f28:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: lhi %r2, 0
; CHECK-NEXT: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32> %a, <4 x i32> %b)
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  %res = and i32 %cc, -4
  ret i32 %res
}

declare <16 x i8> @llvm.s390.vperm(<16 x i8>, <16 x i8>, <16 x i8>)

; Test VPERM (operand elements are 0):
define <16 x i8> @f29() {
; CHECK-LABEL: f29:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: vgbm %v24, 0
; CHECK-NEXT: br %r14
  %perm = call <16 x i8> @llvm.s390.vperm(
                  <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                             i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>,
                  <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                             i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>,
                  <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                             i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>)
  %res = and <16 x i8> %perm, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %res
}

; Test VPERM (operand elements are 1):
define <16 x i8> @f30() {
; CHECK-LABEL: f30:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: vgbm %v0, 0
; CHECK-NEXT: vrepib %v1, 1
; CHECK-NEXT: vperm %v24, %v1, %v1, %v0
; CHECK-NEXT: br %r14
  %perm = call <16 x i8> @llvm.s390.vperm(
                  <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                             i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>,
                  <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                             i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>,
                  <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                             i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>)
  %res = and <16 x i8> %perm, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                               i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %res
}

; Test VPERM with various constant operands.
define i32 @f31() {
; CHECK-LABEL: f31:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT: larl %r1, .LCPI31_0
; CHECK-NEXT: vl %v0, 0(%r1), 3
; CHECK-NEXT: larl %r1, .LCPI31_1
; CHECK-NEXT: vl %v1, 0(%r1), 3
; CHECK-NEXT: vperm %v0, %v1, %v1, %v0
; CHECK-NEXT: vlgvb %r2, %v0, 0
; CHECK-NEXT: nilf %r2, 7
; CHECK-NEXT:                           # kill: def $r2l killed $r2l killed $r2d
; CHECK-NEXT: br %r14
  %P = tail call <16 x i8> @llvm.s390.vperm(<16 x i8> <i8 0, i8 1, i8 1, i8 2, i8 1, i8 2, i8 2, i8 3, i8 1, i8 2, i8 2, i8 3, i8 2, i8 3, i8 3, i8 4>, <16 x i8> <i8 0, i8 1, i8 1, i8 2, i8 1, i8 2, i8 2, i8 3, i8 1, i8 2, i8 2, i8 3, i8 2, i8 3, i8 3, i8 4>, <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0>)
  %E = extractelement <16 x i8> %P, i64 0
  %res = zext i8 %E to i32
  ret i32 %res
}
