; What the prefer-gs-cost-table tuning feature may move: reciprocal throughput
; on an AVX-512 target, and nothing else anywhere. Without AVX-512 it cannot
; reach the rows at all.
;
; Every prefix that may not move is shared between a feature-on and a
; feature-off RUN line, so any movement makes the pair disagree. AVX-512 admits
; the table rather than the vendor, so a non-Zen AVX-512 target is checked too.

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
; a remainder part is the most sensitive case, since a part count taken the wrong
; way shows up there first.
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

; Throughput on Zen is the one kind the feature moves, so these two runs take
; separate prefixes and record the magnitudes.
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=throughput -mcpu=znver4 \
; RUN:   -mattr=-prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=ZEN-TPUT-OFF
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=throughput -mcpu=znver4 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ZEN-TPUT-ON

; An AVX-512 target that is not Zen. Here the feature is not inert, and these
; two runs take separate prefixes to record what it moves.
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=all -mcpu=skylake-avx512 \
; RUN:   -mattr=-prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=SKX-OFF
; RUN: opt < %s -S -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" \
; RUN:   -disable-output -cost-kind=all -mcpu=skylake-avx512 \
; RUN:   -mattr=+prefer-gs-cost-table 2>&1 | FileCheck %s --check-prefix=SKX-ON

; A wider index splits this into two instructions, so the body is charged twice
; while the premium follows the lanes. Only throughput moves, which the shared
; latency and size-latency prefixes hold.
define <16 x i32> @gather_v16i32(<16 x ptr> %ptrs, <16 x i1> %mask) {
; ZEN-TPUT-OFF-LABEL: 'gather_v16i32'
; ZEN-TPUT-OFF: Cost Model: Found an estimated cost of 20 for instruction: %v = call
; ZEN-TPUT-ON-LABEL: 'gather_v16i32'
; ZEN-TPUT-ON: Cost Model: Found an estimated cost of 20 for instruction: %v = call
; ZEN-LAT-LABEL: 'gather_v16i32'
; ZEN-LAT: Cost Model: Found an estimated cost of 68 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v16i32'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 20 for instruction: %v = call
; ZEN-CODESIZE-LABEL: 'gather_v16i32'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 2 for instruction: %v = call
; NOAVX512-LABEL: 'gather_v16i32'
; NOAVX512: Cost Model: Found an estimated cost of 28 for instruction: %v = call
; NOAVX512-CODESIZE-LABEL: 'gather_v16i32'
; NOAVX512-CODESIZE: Cost Model: Found an estimated cost of 4 for instruction: %v = call
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; A length that is not a multiple of its part width, which is where a part count
; taken the wrong way would show first. Without AVX-512 the feature bit stays
; inert on it. On Zen it rounds up to the 16-wide row, so it moves with the bit
; exactly as the 16-wide shape above does.
define <12 x i32> @gather_v12i32(<12 x ptr> %ptrs, <12 x i1> %mask) {
; ZEN-TPUT-OFF-LABEL: 'gather_v12i32'
; ZEN-TPUT-OFF: Cost Model: Found an estimated cost of 16 for instruction: %v = call
; ZEN-TPUT-ON-LABEL: 'gather_v12i32'
; ZEN-TPUT-ON: Cost Model: Found an estimated cost of 16 for instruction: %v = call
; ZEN-LAT-LABEL: 'gather_v12i32'
; ZEN-LAT: Cost Model: Found an estimated cost of 52 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v12i32'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 16 for instruction: %v = call
; ZEN-CODESIZE-LABEL: 'gather_v12i32'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 2 for instruction: %v = call
; NOAVX512-LABEL: 'gather_v12i32'
; NOAVX512: Cost Model: Found an estimated cost of 22 for instruction: %v = call
; NOAVX512-CODESIZE-LABEL: 'gather_v12i32'
; NOAVX512-CODESIZE: Cost Model: Found an estimated cost of 3 for instruction: %v = call
  %v = call <12 x i32> @llvm.masked.gather.v12i32.v12p0(<12 x ptr> %ptrs, i32 4, <12 x i1> %mask, <12 x i32> poison)
  ret <12 x i32> %v
}

define <8 x i32> @gather_v8i32(<8 x ptr> %ptrs, <8 x i1> %mask) {
; ZEN-TPUT-OFF-LABEL: 'gather_v8i32'
; ZEN-TPUT-OFF: Cost Model: Found an estimated cost of 10 for instruction:
; ZEN-TPUT-ON-LABEL: 'gather_v8i32'
; ZEN-TPUT-ON: Cost Model: Found an estimated cost of 10 for instruction:
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
; ZEN-TPUT-OFF-LABEL: 'scatter_v8i64'
; ZEN-TPUT-OFF: Cost Model: Found an estimated cost of 10 for instruction:
; ZEN-TPUT-ON-LABEL: 'scatter_v8i64'
; ZEN-TPUT-ON: Cost Model: Found an estimated cost of 10 for instruction:
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

; The remainder length again, but indexed by a GEP over dwords. Without AVX-512
; the twelve lanes are two instructions and nothing moves with the feature bit.
; With AVX-512 both paths read the dword index and count the one instruction llc
; emits, so the part count and the kinds derived from it agree and only
; throughput moves. znver4 behaves the same and is checked in the index-width
; test.
define <12 x i32> @gather_v12i32_dword_gep(ptr %base, <12 x i32> %idx, <12 x i1> %mask) {
; NOAVX512-ALL-LABEL: 'gather_v12i32_dword_gep'
; NOAVX512-ALL: Cost Model: Found costs of RThru:20 CodeSize:2 Lat:56 SizeLat:20 for: %v = call
; SKX-OFF-LABEL: 'gather_v12i32_dword_gep'
; SKX-OFF: Cost Model: Found costs of RThru:14 CodeSize:1 Lat:50 SizeLat:14 for: %v = call
; SKX-ON-LABEL: 'gather_v12i32_dword_gep'
; SKX-ON: Cost Model: Found costs of RThru:14 CodeSize:1 Lat:50 SizeLat:14 for: %v = call
  %ptrs = getelementptr inbounds i32, ptr %base, <12 x i32> %idx
  %v = call <12 x i32> @llvm.masked.gather.v12i32.v12p0(<12 x ptr> %ptrs, i32 4, <12 x i1> %mask, <12 x i32> poison)
  ret <12 x i32> %v
}

; Legalization widens the index to a fourth register that holds no live lane.
; Both paths count the three parts that do, so the fourth reaches no cost kind
; and the feature moves throughput alone.
define <24 x i32> @gather_v24i32_varmask(<24 x ptr> %ptrs, <24 x i1> %mask) {
; ZEN-TPUT-OFF-LABEL: 'gather_v24i32_varmask'
; ZEN-TPUT-OFF: Cost Model: Found an estimated cost of 34 for instruction: %v = call
; ZEN-TPUT-ON-LABEL: 'gather_v24i32_varmask'
; ZEN-TPUT-ON: Cost Model: Found an estimated cost of 34 for instruction: %v = call
; ZEN-LAT-LABEL: 'gather_v24i32_varmask'
; ZEN-LAT: Cost Model: Found an estimated cost of 106 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v24i32_varmask'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 34 for instruction: %v = call
; ZEN-CODESIZE-LABEL: 'gather_v24i32_varmask'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 3 for instruction: %v = call
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> %mask, <24 x i32> poison)
  ret <24 x i32> %v
}

; A constant mask leaving four of eight lanes live. Both paths charge the body
; over those four and not over the eight the type declares, so the four dead
; lanes reach no cost kind.
define <8 x i32> @gather_v8i32_four_active_lanes(<8 x ptr> %ptrs) {
; ZEN-TPUT-OFF-LABEL: 'gather_v8i32_four_active_lanes'
; ZEN-TPUT-OFF: Cost Model: Found an estimated cost of 6 for instruction: %v = call
; ZEN-TPUT-ON-LABEL: 'gather_v8i32_four_active_lanes'
; ZEN-TPUT-ON: Cost Model: Found an estimated cost of 6 for instruction: %v = call
; ZEN-LAT-LABEL: 'gather_v8i32_four_active_lanes'
; ZEN-LAT: Cost Model: Found an estimated cost of 18 for instruction: %v = call
; ZEN-SIZELAT-LABEL: 'gather_v8i32_four_active_lanes'
; ZEN-SIZELAT: Cost Model: Found an estimated cost of 6 for instruction: %v = call
; ZEN-CODESIZE-LABEL: 'gather_v8i32_four_active_lanes'
; ZEN-CODESIZE: Cost Model: Found an estimated cost of 1 for instruction: %v = call
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> <i1 1, i1 1, i1 1, i1 1, i1 0, i1 0, i1 0, i1 0>, <8 x i32> poison)
  ret <8 x i32> %v
}
