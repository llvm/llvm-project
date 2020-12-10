; Checks if the memory cost model does not break when using scalable vectors

; RUN: opt  -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 8 x i8> @load-sve-8(<vscale x 8 x i8>* %ptr) {
; CHECK-LABEL: 'load-sve-8':
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:
  %retval = load <vscale x 8 x i8>, <vscale x 8 x i8>* %ptr
  ret <vscale x 8 x i8> %retval
}

define void  @store-sve-8(<vscale x 8 x i8>* %ptr, <vscale x 8 x i8> %val) {
; CHECK-LABEL: 'store-sve-8'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:
  store <vscale x 8 x i8> %val, <vscale x 8 x i8>* %ptr
  ret void
}

define <vscale x 16 x i8> @load-sve-16(<vscale x 16 x i8>* %ptr) {
; CHECK-LABEL: 'load-sve-16':
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:
  %retval = load <vscale x 16 x i8>, <vscale x 16 x i8>* %ptr
  ret <vscale x 16 x i8> %retval
}

define void  @store-sve-16(<vscale x 16 x i8>* %ptr, <vscale x 16 x i8> %val) {
; CHECK-LABEL: 'store-sve-16'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:
  store <vscale x 16 x i8> %val, <vscale x 16 x i8>* %ptr
  ret void
}

define <vscale x 32 x i8> @load-sve-32(<vscale x 32 x i8>* %ptr) {
; CHECK-LABEL: 'load-sve-32':
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:
  %retval = load <vscale x 32 x i8>, <vscale x 32 x i8>* %ptr
  ret <vscale x 32 x i8> %retval
}

define void  @store-sve-32(<vscale x 32 x i8>* %ptr, <vscale x 32 x i8> %val) {
; CHECK-LABEL: 'store-sve-32'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:
  store <vscale x 32 x i8> %val, <vscale x 32 x i8>* %ptr
  ret void
}
