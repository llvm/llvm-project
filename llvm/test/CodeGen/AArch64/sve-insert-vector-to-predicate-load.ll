; RUN: opt -S -aarch64-sve-intrinsic-opts < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define <vscale x 16 x i1> @pred_load_v2i8(<2 x i8>* %addr) #0 {
; CHECK-LABEL: @pred_load_v2i8(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <2 x i8>* %addr to <vscale x 16 x i1>*
; CHECK-NEXT:    [[TMP2:%.*]] = load <vscale x 16 x i1>, <vscale x 16 x i1>* [[TMP1]]
; CHECK-NEXT:    ret <vscale x 16 x i1> [[TMP2]]
  %load = load <2 x i8>, <2 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> undef, <2 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

define <vscale x 16 x i1> @pred_load_v4i8(<4 x i8>* %addr) #1 {
; CHECK-LABEL: @pred_load_v4i8(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <4 x i8>* %addr to <vscale x 16 x i1>*
; CHECK-NEXT:    [[TMP2:%.*]] = load <vscale x 16 x i1>, <vscale x 16 x i1>* [[TMP1]]
; CHECK-NEXT:    ret <vscale x 16 x i1> [[TMP2]]
  %load = load <4 x i8>, <4 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v4i8(<vscale x 2 x i8> undef, <4 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

define <vscale x 16 x i1> @pred_load_v8i8(<8 x i8>* %addr) #2 {
; CHECK-LABEL: @pred_load_v8i8(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8>* %addr to <vscale x 16 x i1>*
; CHECK-NEXT:    [[TMP2:%.*]] = load <vscale x 16 x i1>, <vscale x 16 x i1>* [[TMP1]]
; CHECK-NEXT:    ret <vscale x 16 x i1> [[TMP2]]
  %load = load <8 x i8>, <8 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v8i8(<vscale x 2 x i8> undef, <8 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

; Ensure the insertion point is at the load
define <vscale x 16 x i1> @pred_load_insertion_point(<2 x i8>* %addr) #0 {
; CHECK-LABEL: @pred_load_insertion_point(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <2 x i8>* %addr to <vscale x 16 x i1>*
; CHECK-NEXT:    [[TMP2:%.*]] = load <vscale x 16 x i1>, <vscale x 16 x i1>* [[TMP1]]
; CHECK-NEXT:    br label %bb1
; CHECK:       bb1:
; CHECK-NEXT:    ret <vscale x 16 x i1> [[TMP2]]
entry:
  %load = load <2 x i8>, <2 x i8>* %addr, align 4
  br label %bb1

bb1:
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> undef, <2 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

; Check that too small of a vscale prevents optimization
define <vscale x 16 x i1> @pred_load_neg1(<4 x i8>* %addr) #0 {
; CHECK-LABEL: @pred_load_neg1(
; CHECK:         call <vscale x 2 x i8> @llvm.vector.insert
  %load = load <4 x i8>, <4 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v4i8(<vscale x 2 x i8> undef, <4 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

; Check that too large of a vscale prevents optimization
define <vscale x 16 x i1> @pred_load_neg2(<4 x i8>* %addr) #2 {
; CHECK-LABEL: @pred_load_neg2(
; CHECK:         call <vscale x 2 x i8> @llvm.vector.insert
  %load = load <4 x i8>, <4 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v4i8(<vscale x 2 x i8> undef, <4 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

; Check that a non-zero index prevents optimization
define <vscale x 16 x i1> @pred_load_neg3(<4 x i8>* %addr) #1 {
; CHECK-LABEL: @pred_load_neg3(
; CHECK:         call <vscale x 2 x i8> @llvm.vector.insert
  %load = load <4 x i8>, <4 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v4i8(<vscale x 2 x i8> undef, <4 x i8> %load, i64 4)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

; Check that differing vscale min/max prevents optimization
define <vscale x 16 x i1> @pred_load_neg4(<4 x i8>* %addr) #3 {
; CHECK-LABEL: @pred_load_neg4(
; CHECK:         call <vscale x 2 x i8> @llvm.vector.insert
  %load = load <4 x i8>, <4 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v4i8(<vscale x 2 x i8> undef, <4 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

; Check that insertion into a non-undef vector prevents optimization
define <vscale x 16 x i1> @pred_load_neg5(<4 x i8>* %addr, <vscale x 2 x i8> %passthru) #1 {
; CHECK-LABEL: @pred_load_neg5(
; CHECK:         call <vscale x 2 x i8> @llvm.vector.insert
  %load = load <4 x i8>, <4 x i8>* %addr, align 4
  %insert = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v4i8(<vscale x 2 x i8> %passthru, <4 x i8> %load, i64 0)
  %ret = bitcast <vscale x 2 x i8> %insert to <vscale x 16 x i1>
  ret <vscale x 16 x i1> %ret
}

declare <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8>, <2 x i8>, i64)
declare <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v4i8(<vscale x 2 x i8>, <4 x i8>, i64)
declare <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v8i8(<vscale x 2 x i8>, <8 x i8>, i64)

attributes #0 = { "target-features"="+sve" vscale_range(1,1) }
attributes #1 = { "target-features"="+sve" vscale_range(2,2) }
attributes #2 = { "target-features"="+sve" vscale_range(4,4) }
attributes #3 = { "target-features"="+sve" vscale_range(2,4) }
