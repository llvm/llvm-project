; RUN: llc %s -o - | FileCheck %s

target triple = "thumbv7s-apple-ios"

declare <8 x i8> @llvm.arm.neon.vtbl2(<8 x i8> %shuffle.i.i307, <8 x i8> %shuffle.i27.i308, <8 x i8> %vtbl2.i25.i)

; Check that we get the motivating example:
; The bitcasts force the values to go through the GPRs, whereas
; they are defined on VPRs and used on VPRs.
;
; CHECK-LABEL: motivatingExample:
; CHECK: vld1.32 {[[ARG1_VALlo:d[0-9]+]], [[ARG1_VALhi:d[0-9]+]]}, [r0]
; CHECK-NEXT: vldr [[ARG2_VAL:d[0-9]+]], [r1]
; CHECK-NEXT: vtbl.8 [[RES:d[0-9]+]], {[[ARG1_VALlo]], [[ARG1_VALhi]]}, [[ARG2_VAL]]
; CHECK-NEXT: vstr [[RES]], [r1]
; CHECK-NEXT: bx lr
define void @motivatingExample(<2 x i64>* %addr, <8 x i8>* %addr2) {
  %shuffle.i.bc.i309 = load <2 x i64>, <2 x i64>* %addr
  %vtbl2.i25.i = load <8 x i8>, <8 x i8>* %addr2
  %shuffle.i.extract.i310 = extractelement <2 x i64> %shuffle.i.bc.i309, i32 0
  %shuffle.i27.extract.i311 = extractelement <2 x i64> %shuffle.i.bc.i309, i32 1
  %tmp45 = bitcast i64 %shuffle.i.extract.i310 to <8 x i8>
  %tmp46 = bitcast i64 %shuffle.i27.extract.i311 to <8 x i8>
  %vtbl2.i25.i313 = tail call <8 x i8> @llvm.arm.neon.vtbl2(<8 x i8> %tmp45, <8 x i8> %tmp46, <8 x i8> %vtbl2.i25.i)
  store <8 x i8> %vtbl2.i25.i313, <8 x i8>* %addr2
  ret void
}

; Check that we do not perform the transformation for dynamic index.
; CHECK-LABEL: dynamicIndex:
; CHECK-NOT: mul
; CHECK: pop
define void @dynamicIndex(<2 x i64>* %addr, <8 x i8>* %addr2, i32 %index) {
  %shuffle.i.bc.i309 = load <2 x i64>, <2 x i64>* %addr
  %vtbl2.i25.i = load <8 x i8>, <8 x i8>* %addr2
  %shuffle.i.extract.i310 = extractelement <2 x i64> %shuffle.i.bc.i309, i32 %index
  %shuffle.i27.extract.i311 = extractelement <2 x i64> %shuffle.i.bc.i309, i32 1
  %tmp45 = bitcast i64 %shuffle.i.extract.i310 to <8 x i8>
  %tmp46 = bitcast i64 %shuffle.i27.extract.i311 to <8 x i8>
  %vtbl2.i25.i313 = tail call <8 x i8> @llvm.arm.neon.vtbl2(<8 x i8> %tmp45, <8 x i8> %tmp46, <8 x i8> %vtbl2.i25.i)
  store <8 x i8> %vtbl2.i25.i313, <8 x i8>* %addr2
  ret void
}

; Check that we do not perform the transformation when there are several uses
; of the result of the bitcast.
; CHECK-LABEL: severalUses:
; ARG1_VALlo is hard coded because we need to access the high part of d0,
; i.e., s1, and we can't express that with filecheck.
; CHECK: vld1.32 {[[ARG1_VALlo:d0]], [[ARG1_VALhi:d[0-9]+]]}, [r0]
; CHECK-NEXT: vldr [[ARG2_VAL:d[0-9]+]], [r1]
; s1 is actually 2 * ARG1_VALlo + 1, but we cannot express that with filecheck.
; CHECK-NEXT: vmov [[REThi:r[0-9]+]], s1
; We build the return value here. s0 is 2 * ARG1_VALlo.
; CHECK-NEXT: vmov r0, s0
; This copy is correct but actually useless. We should be able to clean it up.
; CHECK-NEXT: vmov [[ARG1_VALloCPY:d[0-9]+]], r0, [[REThi]]
; CHECK-NEXT: vtbl.8 [[RES:d[0-9]+]], {[[ARG1_VALloCPY]], [[ARG1_VALhi]]}, [[ARG2_VAL]]
; CHECK-NEXT: vstr [[RES]], [r1]
; CHECK-NEXT: mov r1, [[REThi]]
; CHECK-NEXT: bx lr
define i64 @severalUses(<2 x i64>* %addr, <8 x i8>* %addr2) {
  %shuffle.i.bc.i309 = load <2 x i64>, <2 x i64>* %addr
  %vtbl2.i25.i = load <8 x i8>, <8 x i8>* %addr2
  %shuffle.i.extract.i310 = extractelement <2 x i64> %shuffle.i.bc.i309, i32 0
  %shuffle.i27.extract.i311 = extractelement <2 x i64> %shuffle.i.bc.i309, i32 1
  %tmp45 = bitcast i64 %shuffle.i.extract.i310 to <8 x i8>
  %tmp46 = bitcast i64 %shuffle.i27.extract.i311 to <8 x i8>
  %vtbl2.i25.i313 = tail call <8 x i8> @llvm.arm.neon.vtbl2(<8 x i8> %tmp45, <8 x i8> %tmp46, <8 x i8> %vtbl2.i25.i)
  store <8 x i8> %vtbl2.i25.i313, <8 x i8>* %addr2
  ret i64 %shuffle.i.extract.i310
}
