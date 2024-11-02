; RUN: opt -S -aarch64-sve-intrinsic-opts < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @pred_store_v2i8(<vscale x 16 x i1> %pred, <2 x i8>* %addr) #0 {
; CHECK-LABEL: @pred_store_v2i8(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <2 x i8>* %addr to <vscale x 16 x i1>*
; CHECK-NEXT:    store <vscale x 16 x i1> %pred, <vscale x 16 x i1>* [[TMP1]]
; CHECK-NEXT:    ret void
  %bitcast = bitcast <vscale x 16 x i1> %pred to <vscale x 2 x i8>
  %extract = tail call <2 x i8> @llvm.vector.extract.v2i8.nxv2i8(<vscale x 2 x i8> %bitcast, i64 0)
  store <2 x i8> %extract, <2 x i8>* %addr, align 4
  ret void
}

define void @pred_store_v4i8(<vscale x 16 x i1> %pred, <4 x i8>* %addr) #1 {
; CHECK-LABEL: @pred_store_v4i8(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <4 x i8>* %addr to <vscale x 16 x i1>*
; CHECK-NEXT:    store <vscale x 16 x i1> %pred, <vscale x 16 x i1>* [[TMP1]]
; CHECK-NEXT:    ret void
  %bitcast = bitcast <vscale x 16 x i1> %pred to <vscale x 2 x i8>
  %extract = tail call <4 x i8> @llvm.vector.extract.v4i8.nxv2i8(<vscale x 2 x i8> %bitcast, i64 0)
  store <4 x i8> %extract, <4 x i8>* %addr, align 4
  ret void
}

define void @pred_store_v8i8(<vscale x 16 x i1> %pred, <8 x i8>* %addr) #2 {
; CHECK-LABEL: @pred_store_v8i8(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8>* %addr to <vscale x 16 x i1>*
; CHECK-NEXT:    store <vscale x 16 x i1> %pred, <vscale x 16 x i1>* [[TMP1]]
; CHECK-NEXT:    ret void
  %bitcast = bitcast <vscale x 16 x i1> %pred to <vscale x 2 x i8>
  %extract = tail call <8 x i8> @llvm.vector.extract.v8i8.nxv2i8(<vscale x 2 x i8> %bitcast, i64 0)
  store <8 x i8> %extract, <8 x i8>* %addr, align 4
  ret void
}


; Check that too small of a vscale prevents optimization
define void @pred_store_neg1(<vscale x 16 x i1> %pred, <4 x i8>* %addr) #0 {
; CHECK-LABEL: @pred_store_neg1(
; CHECK:         call <4 x i8> @llvm.vector.extract
  %bitcast = bitcast <vscale x 16 x i1> %pred to <vscale x 2 x i8>
  %extract = tail call <4 x i8> @llvm.vector.extract.v4i8.nxv2i8(<vscale x 2 x i8> %bitcast, i64 0)
  store <4 x i8> %extract, <4 x i8>* %addr, align 4
  ret void
}

; Check that too large of a vscale prevents optimization
define void @pred_store_neg2(<vscale x 16 x i1> %pred, <4 x i8>* %addr) #2 {
; CHECK-LABEL: @pred_store_neg2(
; CHECK:         call <4 x i8> @llvm.vector.extract
  %bitcast = bitcast <vscale x 16 x i1> %pred to <vscale x 2 x i8>
  %extract = tail call <4 x i8> @llvm.vector.extract.v4i8.nxv2i8(<vscale x 2 x i8> %bitcast, i64 0)
  store <4 x i8> %extract, <4 x i8>* %addr, align 4
  ret void
}

; Check that a non-zero index prevents optimization
define void @pred_store_neg3(<vscale x 16 x i1> %pred, <4 x i8>* %addr) #1 {
; CHECK-LABEL: @pred_store_neg3(
; CHECK:         call <4 x i8> @llvm.vector.extract
  %bitcast = bitcast <vscale x 16 x i1> %pred to <vscale x 2 x i8>
  %extract = tail call <4 x i8> @llvm.vector.extract.v4i8.nxv2i8(<vscale x 2 x i8> %bitcast, i64 4)
  store <4 x i8> %extract, <4 x i8>* %addr, align 4
  ret void
}

; Check that differing vscale min/max prevents optimization
define void @pred_store_neg4(<vscale x 16 x i1> %pred, <4 x i8>* %addr) #3 {
; CHECK-LABEL: @pred_store_neg4(
; CHECK:         call <4 x i8> @llvm.vector.extract
  %bitcast = bitcast <vscale x 16 x i1> %pred to <vscale x 2 x i8>
  %extract = tail call <4 x i8> @llvm.vector.extract.v4i8.nxv2i8(<vscale x 2 x i8> %bitcast, i64 0)
  store <4 x i8> %extract, <4 x i8>* %addr, align 4
  ret void
}

declare <2 x i8> @llvm.vector.extract.v2i8.nxv2i8(<vscale x 2 x i8>, i64)
declare <4 x i8> @llvm.vector.extract.v4i8.nxv2i8(<vscale x 2 x i8>, i64)
declare <8 x i8> @llvm.vector.extract.v8i8.nxv2i8(<vscale x 2 x i8>, i64)

attributes #0 = { "target-features"="+sve" vscale_range(1,1) }
attributes #1 = { "target-features"="+sve" vscale_range(2,2) }
attributes #2 = { "target-features"="+sve" vscale_range(4,4) }
attributes #3 = { "target-features"="+sve" vscale_range(2,4) }
