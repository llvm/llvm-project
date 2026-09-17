; The prefer-gs-cost-table tuning feature must be inert wherever it has nothing
; to say: on a target with no AVX-512, which can never reach the table, and on
; the cost kinds it supplies no numbers for.
;
; Each prefix below is checked by a pair of RUN lines that differ only in the
; feature bit, so if enabling it ever moves one of these costs, the pair
; disagrees and the test fails.
;
; Most shapes here are indexed through vectors of pointers, whose index width is
; not open to question. Where the index comes from a GEP the feature does move
; the part count, and with it the cost kinds derived from that count, because it
; takes the real index width at every VF instead of only at VF 16 and above.
; That difference is deliberate; its magnitudes are pinned in
; masked-gather-scatter-cost-table-index-width.ll.
;
; The last shape is GEP-indexed, and is here rather than there because the two
; halves of "scope" pull apart only on that form. Moving the part count must not
; reach a target the table cannot serve, and must not be confined to Zen either:
; what admits it is AVX-512, not the vendor. So the same GEP shape is checked
; inert on a target without AVX-512, and checked moving on one with AVX-512 that
; is not Zen.

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

; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=code-size -mcpu=znver4 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ZEN-CODESIZE
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=code-size -mcpu=znver4 \
; RUN:   -mattr=-prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=ZEN-CODESIZE

; A target with no AVX-512 can never reach the table, so the feature bit must be
; inert there on every cost kind, not only reciprocal throughput. A length with
; a remainder part is the case that distinguishes the two paths' part counts.
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=throughput -mcpu=skylake 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOAVX512
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=throughput -mcpu=skylake \
; RUN:   -mattr=+prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=NOAVX512

; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=code-size -mcpu=skylake 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOAVX512-CODESIZE
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=code-size -mcpu=skylake \
; RUN:   -mattr=+prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=NOAVX512-CODESIZE

; The same inertness on a target without AVX-512, over every cost kind at once,
; for the GEP-indexed shape. Both runs share a prefix, so any movement fails.
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=all -mcpu=skylake 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOAVX512-ALL
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=all -mcpu=skylake \
; RUN:   -mattr=+prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=NOAVX512-ALL

; An AVX-512 target that is not Zen. Here the feature is not inert, and these
; two runs take separate prefixes to record what it moves.
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=all -mcpu=skylake-avx512 \
; RUN:   -mattr=-prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=SKX-OFF
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=all -mcpu=skylake-avx512 \
; RUN:   -mattr=+prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=SKX-ON

define <16 x i32> @gather_v16i32(<16 x ptr> %ptrs, <16 x i1> %mask) {
; ZEN-LAT-LABEL: 'gather_v16i32'
; ZEN-LAT: Cost Model: Found an estimated cost of 68 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v16i32'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 20 for instruction: %v = call
; ZEN-CODESIZE-LABEL: 'gather_v16i32'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 2 for instruction: %v = call
; NOAVX512-LABEL: 'gather_v16i32'
; NOAVX512: Cost Model: Found an estimated cost of 24 for instruction: %v = call
; NOAVX512-CODESIZE-LABEL: 'gather_v16i32'
; NOAVX512-CODESIZE: Cost Model: Found an estimated cost of 4 for instruction: %v = call
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; A length that is not a multiple of its part width. The Zen path and the
; generic path count parts differently here, so this is the shape that catches
; the feature bit leaking into a target the table cannot serve.
define <12 x i32> @gather_v12i32(<12 x ptr> %ptrs, <12 x i1> %mask) {
; ZEN-LAT-LABEL: 'gather_v12i32'
; ZEN-LAT: Cost Model: Found an estimated cost of 52 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v12i32'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 16 for instruction: %v = call
; ZEN-CODESIZE-LABEL: 'gather_v12i32'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 2 for instruction: %v = call
; NOAVX512-LABEL: 'gather_v12i32'
; NOAVX512: Cost Model: Found an estimated cost of 20 for instruction: %v = call
; NOAVX512-CODESIZE-LABEL: 'gather_v12i32'
; NOAVX512-CODESIZE: Cost Model: Found an estimated cost of 4 for instruction: %v = call
  %v = call <12 x i32> @llvm.masked.gather.v12i32.v12p0(<12 x ptr> %ptrs, i32 4, <12 x i1> %mask, <12 x i32> poison)
  ret <12 x i32> %v
}

define <8 x i32> @gather_v8i32(<8 x ptr> %ptrs, <8 x i1> %mask) {
; ZEN-LAT-LABEL: 'gather_v8i32'
; ZEN-LAT: Cost Model: Found an estimated cost of 34 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v8i32'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 10 for instruction: %v = call
; ZEN-CODESIZE-LABEL: 'gather_v8i32'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 1 for instruction: %v = call
; NOAVX512-LABEL: 'gather_v8i32'
; NOAVX512: Cost Model: Found an estimated cost of 12 for instruction: %v = call
; NOAVX512-CODESIZE-LABEL: 'gather_v8i32'
; NOAVX512-CODESIZE: Cost Model: Found an estimated cost of 2 for instruction: %v = call
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

define void @scatter_v8i64(<8 x i64> %val, <8 x ptr> %ptrs, <8 x i1> %mask) {
; ZEN-LAT-LABEL: 'scatter_v8i64'
; ZEN-LAT: Cost Model: Found an estimated cost of 10 for instruction: call void
; ZEN-SIZELAT-LABEL: 'scatter_v8i64'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 10 for instruction: call void
; ZEN-CODESIZE-LABEL: 'scatter_v8i64'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 1 for instruction: call void
; NOAVX512-LABEL: 'scatter_v8i64'
; NOAVX512: Cost Model: Found an estimated cost of 29 for instruction: call void
; NOAVX512-CODESIZE-LABEL: 'scatter_v8i64'
; NOAVX512-CODESIZE: Cost Model: Found an estimated cost of 37 for instruction: call void
  call void @llvm.masked.scatter.v8i64.v8p0(<8 x i64> %val, <8 x ptr> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

; The remainder length again, but indexed by a GEP over dwords, which is the
; only form that reaches the part count the feature corrects. Without AVX-512
; nothing moves. With it, the twelve lanes are one instruction rather than two,
; and the three cost kinds derived from that count follow the count down; the
; code size of 1 is what llc emits for this shape on both AVX-512 targets, so
; the feature is the accurate side of the pair here, not merely the different
; one. znver4 moves identically and is checked in the index-width test.
define <12 x i32> @gather_v12i32_dword_gep(ptr %base, <12 x i32> %idx, <12 x i1> %mask) {
; NOAVX512-ALL-LABEL: 'gather_v12i32_dword_gep'
; NOAVX512-ALL: Cost Model: Found costs of RThru:20 CodeSize:4 Lat:56 SizeLat:20 for: %v = call
; SKX-OFF-LABEL: 'gather_v12i32_dword_gep'
; SKX-OFF: Cost Model: Found costs of RThru:16 CodeSize:2 Lat:52 SizeLat:16 for: %v = call
; SKX-ON-LABEL: 'gather_v12i32_dword_gep'
; SKX-ON: Cost Model: Found costs of RThru:35 CodeSize:1 Lat:50 SizeLat:14 for: %v = call
  %ptrs = getelementptr inbounds i32, ptr %base, <12 x i32> %idx
  %v = call <12 x i32> @llvm.masked.gather.v12i32.v12p0(<12 x ptr> %ptrs, i32 4, <12 x i1> %mask, <12 x i32> poison)
  ret <12 x i32> %v
}
