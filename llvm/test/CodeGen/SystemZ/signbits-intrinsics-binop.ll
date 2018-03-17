; Test that DAGCombiner gets helped by ComputeNumSignBitsForTargetNode() with
; vector intrinsics.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s

declare {<16 x i8>, i32} @llvm.s390.vpkshs(<8 x i16>, <8 x i16>)
declare {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32>, <4 x i32>)
declare {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64>, <2 x i64>)

; PACKS_CC: i64 -> i32
define <4 x i32> @f0() {
; CHECK-LABEL: f0:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpksgs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64> <i64 0, i64 1>, <2 x i64> <i64 0, i64 1>)
  %extr = extractvalue {<4 x i32>, i32} %call, 0
  %trunc = trunc <4 x i32> %extr to <4 x i16>
  %ret = sext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; PACKS_CC: i32 -> i16
define <8 x i16> @f1() {
; CHECK-LABEL: f1:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpksfs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32> <i32 0, i32 1, i32 1, i32 0>,
                                                  <4 x i32> <i32 0, i32 1, i32 1, i32 0>)
  %extr = extractvalue {<8 x i16>, i32} %call, 0
  %trunc = trunc <8 x i16> %extr to <8 x i8>
  %ret = sext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; PACKS_CC: i16 -> i8
define <16 x i8> @f2() {
; CHECK-LABEL: f2:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpkshs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpkshs(
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>,
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>)
  %extr = extractvalue {<16 x i8>, i32} %call, 0
  %trunc = trunc <16 x i8> %extr to <16 x i4>
  %ret = sext <16 x i4> %trunc to <16 x i8>
  ret <16 x i8> %ret
}

declare {<16 x i8>, i32} @llvm.s390.vpklshs(<8 x i16>, <8 x i16>)
declare {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32>, <4 x i32>)
declare {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64>, <2 x i64>)

; PACKLS_CC: i64 -> i32
define <4 x i32> @f3() {
; CHECK-LABEL: f3:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpklsgs %v24, %v1, %v0
; CHECK-NEXT:  br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64> <i64 0, i64 1>, <2 x i64> <i64 1, i64 0>)
  %extr = extractvalue {<4 x i32>, i32} %call, 0
  %trunc = trunc <4 x i32> %extr to <4 x i16>
  %ret = sext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; PACKLS_CC: i32 -> i16
define <8 x i16> @f4() {
; CHECK-LABEL: f4:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpklsfs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32> <i32 0, i32 1, i32 1, i32 0>,
                                                   <4 x i32> <i32 0, i32 1, i32 1, i32 0>)
  %extr = extractvalue {<8 x i16>, i32} %call, 0
  %trunc = trunc <8 x i16> %extr to <8 x i8>
  %ret = sext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; PACKLS_CC: i16 -> i8
define <16 x i8> @f5() {
; CHECK-LABEL: f5:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpklshs %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpklshs(
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>,
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>)
  %extr = extractvalue {<16 x i8>, i32} %call, 0
  %trunc = trunc <16 x i8> %extr to <16 x i4>
  %ret = sext <16 x i4> %trunc to <16 x i8>
  ret <16 x i8> %ret
}

declare <16 x i8> @llvm.s390.vpksh(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.s390.vpksf(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.s390.vpksg(<2 x i64>, <2 x i64>)

; PACKS: i64 -> i32
define <4 x i32> @f6() {
; CHECK-LABEL: f6:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpksg %v24, %v1, %v0
; CHECK-NEXT:  br %r14
  %call = call <4 x i32> @llvm.s390.vpksg(<2 x i64> <i64 0, i64 1>, <2 x i64> <i64 1, i64 0>)
  %trunc = trunc <4 x i32> %call to <4 x i16>
  %ret = sext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; PACKS: i32 -> i16
define <8 x i16> @f7() {
; CHECK-LABEL: f7:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpksf %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <8 x i16> @llvm.s390.vpksf(<4 x i32> <i32 0, i32 1, i32 1, i32 0>,
                                          <4 x i32> <i32 0, i32 1, i32 1, i32 0>)
  %trunc = trunc <8 x i16> %call to <8 x i8>
  %ret = sext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; PACKS: i16 -> i8
define <16 x i8> @f8() {
; CHECK-LABEL: f8:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpksh %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <16 x i8> @llvm.s390.vpksh(
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>,
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>)
  %trunc = trunc <16 x i8> %call to <16 x i4>
  %ret = sext <16 x i4> %trunc to <16 x i8>
  ret <16 x i8> %ret
}

declare <16 x i8> @llvm.s390.vpklsh(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.s390.vpklsf(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.s390.vpklsg(<2 x i64>, <2 x i64>)

; PACKLS: i64 -> i32
define <4 x i32> @f9() {
; CHECK-LABEL: f9:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpklsg %v24, %v1, %v0
; CHECK-NEXT:  br %r14
  %call = call <4 x i32> @llvm.s390.vpklsg(<2 x i64> <i64 0, i64 1>, <2 x i64> <i64 1, i64 0>)
  %trunc = trunc <4 x i32> %call to <4 x i16>
  %ret = sext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

; PACKLS: i32 -> i16
define <8 x i16> @f10() {
; CHECK-LABEL: f10:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpklsf %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <8 x i16> @llvm.s390.vpklsf(<4 x i32> <i32 0, i32 1, i32 1, i32 0>,
                                           <4 x i32> <i32 0, i32 1, i32 1, i32 0>)
  %trunc = trunc <8 x i16> %call to <8 x i8>
  %ret = sext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

; PACKLS: i16 -> i8
define <16 x i8> @f11() {
; CHECK-LABEL: f11:
; CHECK-LABEL: # %bb.0:
; CHECK:       vpklsh %v24, %v0, %v0
; CHECK-NEXT:  br %r14
  %call = call <16 x i8> @llvm.s390.vpklsh(
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>,
                <8 x i16> <i16 0, i16 0, i16 1, i16 1, i16 0, i16 0, i16 1, i16 1>)
  %trunc = trunc <16 x i8> %call to <16 x i4>
  %ret = sext <16 x i4> %trunc to <16 x i8>
  ret <16 x i8> %ret
}

declare <2 x i64> @llvm.s390.vpdi(<2 x i64>, <2 x i64>, i32)

; VPDI:
define <2 x i64> @f12() {
; CHECK-LABEL: f12:
; CHECK-LABEL: # %bb.0:
; CHECK:      vpdi %v24, %v1, %v0, 0
; CHECK-NEXT: br %r14
  %perm = call <2 x i64> @llvm.s390.vpdi(<2 x i64> <i64 0, i64 1>,
                                         <2 x i64> <i64 1, i64 0>, i32 0)
  %trunc = trunc <2 x i64> %perm to <2 x i32>
  %ret = sext <2 x i32> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

declare <16 x i8> @llvm.s390.vsldb(<16 x i8>, <16 x i8>, i32)

; VSLDB:
define <16 x i8> @f13() {
; CHECK-LABEL: f13:
; CHECK-LABEL: # %bb.0:
; CHECK:      vsldb %v24, %v0, %v0, 1
; CHECK-NEXT: br %r14
  %shfd = call <16 x i8> @llvm.s390.vsldb(<16 x i8>
                 <i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1,
                  i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1>, <16 x i8>
                 <i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1,
                  i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1>,
                  i32 1)
  %trunc = trunc <16 x i8> %shfd to <16 x i4>
  %ret = sext <16 x i4> %trunc to <16 x i8>
  ret <16 x i8> %ret
}

declare <16 x i8> @llvm.s390.vperm(<16 x i8>, <16 x i8>, <16 x i8>)

; Test VPERM:
define <16 x i8> @f14() {
; CHECK-LABEL: f14:
; CHECK-LABEL: # %bb.0:
; CHECK:      vperm %v24, %v0, %v0, %v0
; CHECK-NEXT: br %r14
  %perm = call <16 x i8> @llvm.s390.vperm(
                  <16 x i8> <i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1,
                             i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1>,
                  <16 x i8> <i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1,
                             i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1>,
                  <16 x i8> <i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1,
                             i8 0, i8 0, i8 1, i8 1, i8 0, i8 1, i8 1, i8 1>)
  %trunc = trunc <16 x i8> %perm to <16 x i4>
  %ret = sext <16 x i4> %trunc to <16 x i8>
  ret <16 x i8> %ret
}
