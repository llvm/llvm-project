; Check memory cost model action for fixed vector SVE and Neon
; Vector bits size lower than 256 bits end up assuming Neon cost model
; CHECK-NEON has same performance as CHECK-SVE-128

; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+neon  < %s | FileCheck %s --check-prefix=CHECK-NEON
; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=128 < %s | FileCheck %s --check-prefix=CHECK-SVE-128
; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=256 < %s | FileCheck %s --check-prefix=CHECK-SVE-256
; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=512 < %s | FileCheck %s --check-prefix=CHECK-SVE-512

define <16 x i8> @load16(<16 x i8>* %ptr) {
; CHECK: 'Cost Model Analysis' for function 'load16':
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <16 x i8>, <16 x i8>* %ptr
  ret <16 x i8> %out
}

define void @store16(<16 x i8>* %ptr, <16 x i8> %val) {
; CHECK: 'Cost Model Analysis' for function 'store16':
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  store <16 x i8> %val, <16 x i8>* %ptr
  ret void
}

define <8 x i8> @load8(<8 x i8>* %ptr) {
; CHECK: 'Cost Model Analysis' for function 'load8':
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <8 x i8>, <8 x i8>* %ptr
  ret <8 x i8> %out
}

define void @store8(<8 x i8>* %ptr, <8 x i8> %val) {
; CHECK: 'Cost Model Analysis' for function 'store8':
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  store <8 x i8> %val, <8 x i8>* %ptr
  ret void
}

define <4 x i8> @load4(<4 x i8>* %ptr) {
; CHECK: 'Cost Model Analysis' for function 'load4':
; CHECK-NEON: Cost Model: Found an estimated cost of 64 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 64 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <4 x i8>, <4 x i8>* %ptr
  ret <4 x i8> %out
}

define void @store4(<4 x i8>* %ptr, <4 x i8> %val) {
; CHECK: 'Cost Model Analysis' for function 'store4':
; CHECK-NEON: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  store <4 x i8> %val, <4 x i8>* %ptr
  ret void
}

define <16 x i16> @load_256(<16 x i16>* %ptr) {
; CHECK: 'Cost Model Analysis' for function 'load_256':
; CHECK-NEON: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 1 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <16 x i16>, <16 x i16>* %ptr
  ret <16 x i16> %out
}

define <8 x i64> @load_512(<8 x i64>* %ptr) {
; CHECK: 'Cost Model Analysis' for function 'load_512':
; CHECK-NEON: Cost Model: Found an estimated cost of 4 for instruction:
; CHECK-SVE-128: Cost Model: Found an estimated cost of 4 for instruction:
; CHECK-SVE-256: Cost Model: Found an estimated cost of 2 for instruction:
; CHECK-SVE-512: Cost Model: Found an estimated cost of 1 for instruction:
  %out = load <8 x i64>, <8 x i64>* %ptr
  ret <8 x i64> %out
}
