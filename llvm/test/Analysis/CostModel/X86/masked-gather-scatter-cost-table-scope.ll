; The prefer-gs-cost-table tuning feature supplies reciprocal-throughput numbers
; only, and only for AVX-512 shapes it has a row for. Enabling it must therefore
; not move any other cost kind, and must not affect a target the table does not
; cover at all.
;
; Each prefix below is checked by a pair of RUN lines that differ only in the
; feature bit, so if enabling it ever moves one of these costs, the pair
; disagrees and the test fails.

; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=latency -mcpu=znver4 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ZEN-LAT
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=latency -mcpu=znver4 \
; RUN:   -mattr=-prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=ZEN-LAT

; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=size-latency -mcpu=znver4 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ZEN-SIZELAT
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=size-latency -mcpu=znver4 \
; RUN:   -mattr=-prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=ZEN-SIZELAT

; A target with no AVX-512 can never reach the table, so the feature bit must be
; inert there even for reciprocal throughput.
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=throughput -mcpu=skylake 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOAVX512
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=throughput -mcpu=skylake \
; RUN:   -mattr=+prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=NOAVX512

define <16 x i32> @gather_v16i32(<16 x ptr> %ptrs, <16 x i1> %mask) {
; ZEN-LAT-LABEL: 'gather_v16i32'
; ZEN-LAT: Cost Model: Found an estimated cost of 68 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v16i32'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 20 for instruction: %v = call
; NOAVX512-LABEL: 'gather_v16i32'
; NOAVX512: Cost Model: Found an estimated cost of 24 for instruction: %v = call
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

define <8 x i32> @gather_v8i32(<8 x ptr> %ptrs, <8 x i1> %mask) {
; ZEN-LAT-LABEL: 'gather_v8i32'
; ZEN-LAT: Cost Model: Found an estimated cost of 34 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v8i32'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 10 for instruction: %v = call
; NOAVX512-LABEL: 'gather_v8i32'
; NOAVX512: Cost Model: Found an estimated cost of 12 for instruction: %v = call
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

define void @scatter_v8i64(<8 x i64> %val, <8 x ptr> %ptrs, <8 x i1> %mask) {
; ZEN-LAT-LABEL: 'scatter_v8i64'
; ZEN-LAT: Cost Model: Found an estimated cost of 10 for instruction: call void
; ZEN-SIZELAT-LABEL: 'scatter_v8i64'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 10 for instruction: call void
; NOAVX512-LABEL: 'scatter_v8i64'
; NOAVX512: Cost Model: Found an estimated cost of 29 for instruction: call void
  call void @llvm.masked.scatter.v8i64.v8p0(<8 x i64> %val, <8 x ptr> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}
