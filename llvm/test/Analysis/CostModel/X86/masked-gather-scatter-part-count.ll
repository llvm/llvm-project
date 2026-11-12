; Pins the cost model's part count against the number of gather/scatter
; instructions CodeGen actually emits, for both index widths on an AVX-512 and
; an AVX2 target.
;
; The pairing is the point: the code-size cost of a hardware gather/scatter is
; its part count, so every cost check below has a matching instruction count
; taken from the same module. The throughput run pins the per-lane term, which
; code size does not expose.
;
; Maintained by hand rather than by update_analyze_test_checks.py, which does
; not know about the paired llc runs.

; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=code-size -disable-output -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefix=SKX-COST
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=throughput -disable-output -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefix=SKX-TPUT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 | FileCheck %s --check-prefix=SKX-ASM
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=code-size -disable-output -mcpu=skylake 2>&1 | FileCheck %s --check-prefix=AVX2-COST
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake | FileCheck %s --check-prefix=AVX2-ASM
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=code-size -disable-output -mcpu=knl 2>&1 | FileCheck %s --check-prefix=KNL-COST
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=knl | FileCheck %s --check-prefix=KNL-ASM
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=code-size -disable-output -mcpu=skylake-avx512 -mattr=-avx512bw 2>&1 | FileCheck %s --check-prefix=NOBW-COST
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 -mattr=-avx512bw | FileCheck %s --check-prefix=NOBW-ASM

; A length that fills its single part exactly, to sit against the nine-lane form
; below: both are one instruction, but the ninth lane is still accessed and has
; to be charged, so the throughput costs must differ by exactly one lane.
; SKX-COST-LABEL: 'gather_v8i32_dword_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v8i32_dword_index'
; SKX-TPUT: cost of 10 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8i32_dword_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
define <8 x i32> @gather_v8i32_dword_index(ptr %base, <8 x i32> %idx, <8 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <8 x i32> %idx
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

; A vector length that is not a multiple of its part count: the remainder lane
; still needs a part of its own once the index no longer fits alongside the rest.
; SKX-COST-LABEL: 'gather_v9i32_dword_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v9i32_dword_index'
; SKX-TPUT: cost of 11 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v9i32_dword_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v9i32_dword_index'
; AVX2-COST: cost of 6 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v9i32_dword_index:
; AVX2-ASM-COUNT-2: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <9 x i32> @gather_v9i32_dword_index(ptr %base, <9 x i32> %idx, <9 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <9 x i32> %idx
  %v = call <9 x i32> @llvm.masked.gather.v9i32.v9p0(<9 x ptr> %ptrs, i32 4, <9 x i1> %mask, <9 x i32> poison)
  ret <9 x i32> %v
}

; SKX-COST-LABEL: 'gather_v9i32_qword_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v9i32_qword_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v9i32_qword_index'
; AVX2-COST: cost of 7 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v9i32_qword_index:
; AVX2-ASM-COUNT-3: vpgatherqd
; AVX2-ASM-NOT: vpgather
define <9 x i32> @gather_v9i32_qword_index(ptr %base, <9 x i64> %idx, <9 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <9 x i64> %idx
  %v = call <9 x i32> @llvm.masked.gather.v9i32.v9p0(<9 x ptr> %ptrs, i32 4, <9 x i1> %mask, <9 x i32> poison)
  ret <9 x i32> %v
}

; The same shape indexed more widely than a pointer, which is not a narrow
; index: it is cut down to pointer width rather than to a dword, so it costs
; what the qword form above costs. Asking whether the index type is exactly 64
; bits wide would miss it and price it as a dword.
; SKX-COST-LABEL: 'gather_v9i32_oversized_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v9i32_oversized_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v9i32_oversized_index'
; AVX2-COST: cost of 7 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v9i32_oversized_index:
; AVX2-ASM-COUNT-3: vpgatherqd
; AVX2-ASM-NOT: vpgatherqd
define <9 x i32> @gather_v9i32_oversized_index(ptr %base, <9 x i128> %idx, <9 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <9 x i128> %idx
  %v = call <9 x i32> @llvm.masked.gather.v9i32.v9p0(<9 x ptr> %ptrs, i32 4, <9 x i1> %mask, <9 x i32> poison)
  ret <9 x i32> %v
}

; A length below the width at which the index used to be examined, so both index
; widths would otherwise be priced the same on either target.
; SKX-COST-LABEL: 'gather_v17i32_dword_index'
; SKX-COST: cost of 6 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v17i32_dword_index:
; SKX-ASM-COUNT-2: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v17i32_dword_index'
; AVX2-COST: cost of 11 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v17i32_dword_index:
; AVX2-ASM-COUNT-3: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <17 x i32> @gather_v17i32_dword_index(ptr %base, <17 x i32> %idx, <17 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <17 x i32> %idx
  %v = call <17 x i32> @llvm.masked.gather.v17i32.v17p0(<17 x ptr> %ptrs, i32 4, <17 x i1> %mask, <17 x i32> poison)
  ret <17 x i32> %v
}

; SKX-COST-LABEL: 'gather_v17i32_qword_index'
; SKX-COST: cost of 7 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v17i32_qword_index:
; SKX-ASM-COUNT-3: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v17i32_qword_index'
; AVX2-COST: cost of 13 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v17i32_qword_index:
; AVX2-ASM-COUNT-5: vpgatherqd
; AVX2-ASM-NOT: vpgather
define <17 x i32> @gather_v17i32_qword_index(ptr %base, <17 x i64> %idx, <17 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <17 x i64> %idx
  %v = call <17 x i32> @llvm.masked.gather.v17i32.v17p0(<17 x ptr> %ptrs, i32 4, <17 x i1> %mask, <17 x i32> poison)
  ret <17 x i32> %v
}

; A length whose qword indices occupy four legal registers but fill only three
; of them with live lanes.
; SKX-COST-LABEL: 'gather_v24i32_dword_index'
; SKX-COST: cost of 6 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_dword_index:
; SKX-ASM-COUNT-2: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v24i32_dword_index'
; AVX2-COST: cost of 11 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v24i32_dword_index:
; AVX2-ASM-COUNT-3: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <24 x i32> @gather_v24i32_dword_index(ptr %base, <24 x i32> %idx, <24 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i32> %idx
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> %mask, <24 x i32> poison)
  ret <24 x i32> %v
}

; SKX-COST-LABEL: 'gather_v24i32_qword_index'
; SKX-COST: cost of 7 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v24i32_qword_index'
; SKX-TPUT: cost of 34 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_qword_index:
; SKX-ASM-COUNT-3: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v24i32_qword_index'
; AVX2-COST: cost of 14 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v24i32_qword_index:
; AVX2-ASM-COUNT-6: vpgatherqd
; AVX2-ASM-NOT: vpgather
define <24 x i32> @gather_v24i32_qword_index(ptr %base, <24 x i64> %idx, <24 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> %mask, <24 x i32> poison)
  ret <24 x i32> %v
}

; Scatters with a variable mask. AVX2 has no hardware scatter, so its cost comes
; from scalarizing instead of from a part count, and it emits no scatter at all.
; SKX-COST-LABEL: 'scatter_v9i32_dword_index_var_mask'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v9i32_dword_index_var_mask:
; SKX-ASM-COUNT-1: vpscatterdd
; SKX-ASM-NOT: vpscatter
; AVX2-ASM-LABEL: scatter_v9i32_dword_index_var_mask:
; AVX2-ASM-NOT: vpscatter
define void @scatter_v9i32_dword_index_var_mask(ptr %base, <9 x i32> %idx, <9 x i32> %val, <9 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <9 x i32> %idx
  call void @llvm.masked.scatter.v9i32.v9p0(<9 x i32> %val, <9 x ptr> %ptrs, i32 4, <9 x i1> %mask)
  ret void
}

; SKX-COST-LABEL: 'scatter_v9i32_qword_index_var_mask'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v9i32_qword_index_var_mask:
; SKX-ASM-COUNT-2: vpscatterqd
; SKX-ASM-NOT: vpscatter
; AVX2-ASM-LABEL: scatter_v9i32_qword_index_var_mask:
; AVX2-ASM-NOT: vpscatter
define void @scatter_v9i32_qword_index_var_mask(ptr %base, <9 x i64> %idx, <9 x i32> %val, <9 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <9 x i64> %idx
  call void @llvm.masked.scatter.v9i32.v9p0(<9 x i32> %val, <9 x ptr> %ptrs, i32 4, <9 x i1> %mask)
  ret void
}

; Scatters with every lane active: an all-ones mask leaves the part count alone.
; SKX-COST-LABEL: 'scatter_v24i32_dword_index_all_active'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v24i32_dword_index_all_active:
; SKX-ASM-COUNT-2: vpscatterdd
; SKX-ASM-NOT: vpscatter
; AVX2-ASM-LABEL: scatter_v24i32_dword_index_all_active:
; AVX2-ASM-NOT: vpscatter
define void @scatter_v24i32_dword_index_all_active(ptr %base, <24 x i32> %idx, <24 x i32> %val) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i32> %idx
  call void @llvm.masked.scatter.v24i32.v24p0(<24 x i32> %val, <24 x ptr> %ptrs, i32 4, <24 x i1> splat (i1 true))
  ret void
}

; SKX-COST-LABEL: 'scatter_v24i32_qword_index_all_active'
; SKX-COST: cost of 3 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v24i32_qword_index_all_active:
; SKX-ASM-COUNT-3: vpscatterqd
; SKX-ASM-NOT: vpscatter
; AVX2-ASM-LABEL: scatter_v24i32_qword_index_all_active:
; AVX2-ASM-NOT: vpscatter
define void @scatter_v24i32_qword_index_all_active(ptr %base, <24 x i64> %idx, <24 x i32> %val) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  call void @llvm.masked.scatter.v24i32.v24p0(<24 x i32> %val, <24 x ptr> %ptrs, i32 4, <24 x i1> splat (i1 true))
  ret void
}

; A variable mask on a length that legalization widens past a part boundary. The
; widened tail cannot be proved dead, and it survives as a store under a zeroed
; mask, so CodeGen emits a fourth instruction where the all-ones form above
; emits three. That store is counted, since it is still an instruction.
; SKX-COST-LABEL: 'scatter_v24i32_qword_index_var_mask'
; SKX-COST: cost of 8 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v24i32_qword_index_var_mask:
; SKX-ASM-COUNT-3: vpscatterqd
; SKX-ASM: kxor
; SKX-ASM-COUNT-1: vpscatterqd
; SKX-ASM-NOT: vpscatter
; AVX2-ASM-LABEL: scatter_v24i32_qword_index_var_mask:
; AVX2-ASM-NOT: vpscatter
define void @scatter_v24i32_qword_index_var_mask(ptr %base, <24 x i64> %idx, <24 x i32> %val, <24 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  call void @llvm.masked.scatter.v24i32.v24p0(<24 x i32> %val, <24 x ptr> %ptrs, i32 4, <24 x i1> %mask)
  ret void
}

; A compile-time mask says which parts survive. With only the first eight lanes
; live, two of these three parts hold nothing and are folded away, leaving one
; instruction rather than the three the declared length would suggest.
; The dead lanes are not charged either: against the 30 the same shape costs
; with an unknown mask above, this is the cost of the eight lanes it reaches.
; SKX-COST-LABEL: 'gather_v24i32_qword_index_first8_mask'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v24i32_qword_index_first8_mask'
; SKX-TPUT: cost of 10 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_qword_index_first8_mask:
; SKX-ASM-COUNT-1: vpgatherqd
; SKX-ASM-NOT: vpgather
define <24 x i32> @gather_v24i32_qword_index_first8_mask(ptr %base, <24 x i64> %idx) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <24 x i32> poison)
  ret <24 x i32> %v
}

; The same for a scatter, where the dead parts would otherwise be the zeroed
; stores counted above.
; SKX-COST-LABEL: 'scatter_v24i32_qword_index_first8_mask'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v24i32_qword_index_first8_mask:
; SKX-ASM-COUNT-1: vpscatterqd
; SKX-ASM-NOT: vpscatter
define void @scatter_v24i32_qword_index_first8_mask(ptr %base, <24 x i64> %idx, <24 x i32> %val) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  call void @llvm.masked.scatter.v24i32.v24p0(<24 x i32> %val, <24 x ptr> %ptrs, i32 4, <24 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>)
  ret void
}

; A mask with no live lane leaves nothing to do, and CodeGen emits no gather at
; all, so the operation is free rather than costed as three parts.
; SKX-COST-LABEL: 'gather_v24i32_qword_index_zero_mask'
; SKX-COST: cost of 0 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v24i32_qword_index_zero_mask'
; SKX-TPUT: cost of 0 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_qword_index_zero_mask:
; SKX-ASM-NOT: vpgather
define <24 x i32> @gather_v24i32_qword_index_zero_mask(ptr %base, <24 x i64> %idx) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> zeroinitializer, <24 x i32> poison)
  ret <24 x i32> %v
}

; The same for a scatter, which unlike a gather would otherwise still be charged
; for the tail it emits under a zeroed mask.
; SKX-COST-LABEL: 'scatter_v24i32_qword_index_zero_mask'
; SKX-COST: cost of 0 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v24i32_qword_index_zero_mask:
; SKX-ASM-NOT: vpscatter
define void @scatter_v24i32_qword_index_zero_mask(ptr %base, <24 x i64> %idx, <24 x i32> %val) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  call void @llvm.masked.scatter.v24i32.v24p0(<24 x i32> %val, <24 x ptr> %ptrs, i32 4, <24 x i1> zeroinitializer)
  ret void
}

; The live lanes need not be a prefix. Here they sit in the first and last part
; with a dead one between, so it is the parts holding a live lane that are
; counted, not the number of live lanes rounded up to a part.
; SKX-COST-LABEL: 'gather_v24i32_qword_index_gapped_mask'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v24i32_qword_index_gapped_mask'
; SKX-TPUT: cost of 6 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_qword_index_gapped_mask:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
define <24 x i32> @gather_v24i32_qword_index_gapped_mask(ptr %base, <24 x i64> %idx) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> <i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <24 x i32> poison)
  ret <24 x i32> %v
}

; Live lanes confined to the last part, the mirror of the first-eight case, so
; that a part is not counted merely because earlier parts were.
; SKX-COST-LABEL: 'gather_v24i32_qword_index_last8_mask'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_qword_index_last8_mask:
; SKX-ASM-COUNT-1: vpgatherqd
; SKX-ASM-NOT: vpgather
define <24 x i32> @gather_v24i32_qword_index_last8_mask(ptr %base, <24 x i64> %idx) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i64> %idx
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> <i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <24 x i32> poison)
  ret <24 x i32> %v
}

; How far that widened tail reaches is set by the predicate rather than by the
; data. The mask legalizes into k-register pieces, 32 lanes wide with AVX512BW
; and 16 without when zmm is in use, and the length is the next whole number of
; pieces. So 48 lanes reach 64 with the wide predicate and eight parts are
; emitted, and stay at 48 without it, where six are. Counting legal registers
; would give eight on both.
; SKX-COST-LABEL: 'scatter_v48i32_qword_index_var_mask'
; SKX-COST: cost of 16 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v48i32_qword_index_var_mask:
; SKX-ASM-COUNT-8: vpscatterqd
; SKX-ASM-NOT: vpscatter
; KNL-COST-LABEL: 'scatter_v48i32_qword_index_var_mask'
; KNL-COST: cost of 6 for instruction: {{.*}}masked.scatter
; KNL-ASM-LABEL: scatter_v48i32_qword_index_var_mask:
; KNL-ASM-COUNT-6: vpscatterqd
; KNL-ASM-NOT: vpscatter
define void @scatter_v48i32_qword_index_var_mask(ptr %base, <48 x i64> %idx, <48 x i32> %val, <48 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <48 x i64> %idx
  call void @llvm.masked.scatter.v48i32.v48p0(<48 x i32> %val, <48 x ptr> %ptrs, i32 4, <48 x i1> %mask)
  ret void
}

; The same split with dword indices, where a part covers sixteen lanes instead
; of eight: four parts with the wide predicate, three without it.
; SKX-COST-LABEL: 'scatter_v48i32_dword_index_var_mask'
; SKX-COST: cost of 12 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v48i32_dword_index_var_mask:
; SKX-ASM-COUNT-4: vpscatterdd
; SKX-ASM-NOT: vpscatter
; KNL-COST-LABEL: 'scatter_v48i32_dword_index_var_mask'
; KNL-COST: cost of 3 for instruction: {{.*}}masked.scatter
; KNL-ASM-LABEL: scatter_v48i32_dword_index_var_mask:
; KNL-ASM-COUNT-3: vpscatterdd
; KNL-ASM-NOT: vpscatter
define void @scatter_v48i32_dword_index_var_mask(ptr %base, <48 x i32> %idx, <48 x i32> %val, <48 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <48 x i32> %idx
  call void @llvm.masked.scatter.v48i32.v48p0(<48 x i32> %val, <48 x ptr> %ptrs, i32 4, <48 x i1> %mask)
  ret void
}

; Past 64 lanes the widened length is no longer a power of two under either
; predicate: 65 reaches 96 in 32-lane pieces and 80 in 16-lane ones, for twelve
; parts and ten. Rounding the length up to a power of two instead would reach
; 128 and overcharge both.
; SKX-COST-LABEL: 'scatter_v65i32_qword_index_var_mask'
; SKX-COST: cost of 28 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v65i32_qword_index_var_mask:
; SKX-ASM-COUNT-12: vpscatterqd
; SKX-ASM-NOT: vpscatter
; KNL-COST-LABEL: 'scatter_v65i32_qword_index_var_mask'
; KNL-COST: cost of 10 for instruction: {{.*}}masked.scatter
; KNL-ASM-LABEL: scatter_v65i32_qword_index_var_mask:
; KNL-ASM-COUNT-10: vpscatterqd
; KNL-ASM-NOT: vpscatter
define void @scatter_v65i32_qword_index_var_mask(ptr %base, <65 x i64> %idx, <65 x i32> %val, <65 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <65 x i64> %idx
  call void @llvm.masked.scatter.v65i32.v65p0(<65 x i32> %val, <65 x ptr> %ptrs, i32 4, <65 x i1> %mask)
  ret void
}

; The same length with dword indices, sixteen lanes to a part: six and five.
; These deltas differ from the qword pair, so no constant satisfies both.
; SKX-COST-LABEL: 'scatter_v65i32_dword_index_var_mask'
; SKX-COST: cost of 20 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v65i32_dword_index_var_mask:
; SKX-ASM-COUNT-6: vpscatterdd
; SKX-ASM-NOT: vpscatter
; KNL-COST-LABEL: 'scatter_v65i32_dword_index_var_mask'
; KNL-COST: cost of 5 for instruction: {{.*}}masked.scatter
; KNL-ASM-LABEL: scatter_v65i32_dword_index_var_mask:
; KNL-ASM-COUNT-5: vpscatterdd
; KNL-ASM-NOT: vpscatter
define void @scatter_v65i32_dword_index_var_mask(ptr %base, <65 x i32> %idx, <65 x i32> %val, <65 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <65 x i32> %idx
  call void @llvm.masked.scatter.v65i32.v65p0(<65 x i32> %val, <65 x ptr> %ptrs, i32 4, <65 x i1> %mask)
  ret void
}

; How wide a piece is does not follow from AVX512BW alone. Sixteen 64-bit values
; need two zmm to hold, so with no zmm the piece stops at eight. All three
; answers below are the same 17 lanes with a qword index: 32 lanes in 32-lane
; pieces, 24 in 8-lane ones, and 32 again on a target whose lack of VLX keeps
; zmm in use whatever the function asks for.
; SKX-COST-LABEL: 'scatter_v17i32_qword_index_var_mask_256bit'
; SKX-COST: cost of 16 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v17i32_qword_index_var_mask_256bit:
; SKX-ASM-COUNT-8: vpscatterqd
; SKX-ASM-NOT: vpscatter
; NOBW-COST-LABEL: 'scatter_v17i32_qword_index_var_mask_256bit'
; NOBW-COST: cost of 12 for instruction: {{.*}}masked.scatter
; NOBW-ASM-LABEL: scatter_v17i32_qword_index_var_mask_256bit:
; NOBW-ASM-COUNT-6: vpscatterqd
; NOBW-ASM-NOT: vpscatter
; KNL-COST-LABEL: 'scatter_v17i32_qword_index_var_mask_256bit'
; KNL-COST: cost of 4 for instruction: {{.*}}masked.scatter
; KNL-ASM-LABEL: scatter_v17i32_qword_index_var_mask_256bit:
; KNL-ASM-COUNT-4: vpscatterqd
; KNL-ASM-NOT: vpscatter
define void @scatter_v17i32_qword_index_var_mask_256bit(ptr %base, <17 x i64> %idx, <17 x i32> %val, <17 x i1> %mask) #0 {
  %ptrs = getelementptr inbounds i32, ptr %base, <17 x i64> %idx
  call void @llvm.masked.scatter.v17i32.v17p0(<17 x i32> %val, <17 x ptr> %ptrs, i32 4, <17 x i1> %mask)
  ret void
}

; A dword index on dword data fits sixteen lanes in one zmm either way, so
; dropping zmm does not narrow the piece: the two answers agree at four parts.
; Halving the piece for this shape too would give three.
; SKX-COST-LABEL: 'scatter_v17i32_dword_index_var_mask_256bit'
; SKX-COST: cost of 12 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v17i32_dword_index_var_mask_256bit:
; SKX-ASM-COUNT-4: vpscatterdd
; SKX-ASM-NOT: vpscatter
; NOBW-COST-LABEL: 'scatter_v17i32_dword_index_var_mask_256bit'
; NOBW-COST: cost of 8 for instruction: {{.*}}masked.scatter
; NOBW-ASM-LABEL: scatter_v17i32_dword_index_var_mask_256bit:
; NOBW-ASM-COUNT-4: vpscatterdd
; NOBW-ASM-NOT: vpscatter
; KNL-COST-LABEL: 'scatter_v17i32_dword_index_var_mask_256bit'
; KNL-COST: cost of 2 for instruction: {{.*}}masked.scatter
; KNL-ASM-LABEL: scatter_v17i32_dword_index_var_mask_256bit:
; KNL-ASM-COUNT-2: vpscatterdd
; KNL-ASM-NOT: vpscatter
define void @scatter_v17i32_dword_index_var_mask_256bit(ptr %base, <17 x i32> %idx, <17 x i32> %val, <17 x i1> %mask) #0 {
  %ptrs = getelementptr inbounds i32, ptr %base, <17 x i32> %idx
  call void @llvm.masked.scatter.v17i32.v17p0(<17 x i32> %val, <17 x ptr> %ptrs, i32 4, <17 x i1> %mask)
  ret void
}

; It is the wider of the data and the index that sets the piece, not the index
; alone. A dword index over 64-bit data still puts only four lanes in a 256-bit
; part, so the piece halves here exactly as it does for a qword index, and the
; two narrow answers match the qword pair above rather than the dword one.
; SKX-COST-LABEL: 'scatter_v17i64_dword_index_var_mask_256bit'
; SKX-COST: cost of 24 for instruction: {{.*}}masked.scatter
; SKX-ASM-LABEL: scatter_v17i64_dword_index_var_mask_256bit:
; SKX-ASM-COUNT-8: vpscatterdq
; SKX-ASM-NOT: vpscatter
; NOBW-COST-LABEL: 'scatter_v17i64_dword_index_var_mask_256bit'
; NOBW-COST: cost of 20 for instruction: {{.*}}masked.scatter
; NOBW-ASM-LABEL: scatter_v17i64_dword_index_var_mask_256bit:
; NOBW-ASM-COUNT-6: vpscatterdq
; NOBW-ASM-NOT: vpscatter
; KNL-COST-LABEL: 'scatter_v17i64_dword_index_var_mask_256bit'
; KNL-COST: cost of 8 for instruction: {{.*}}masked.scatter
; KNL-ASM-LABEL: scatter_v17i64_dword_index_var_mask_256bit:
; KNL-ASM-COUNT-4: vpscatterdq
; KNL-ASM-NOT: vpscatter
define void @scatter_v17i64_dword_index_var_mask_256bit(ptr %base, <17 x i32> %idx, <17 x i64> %val, <17 x i1> %mask) #0 {
  %ptrs = getelementptr inbounds i64, ptr %base, <17 x i32> %idx
  call void @llvm.masked.scatter.v17i64.v17p0(<17 x i64> %val, <17 x ptr> %ptrs, i32 8, <17 x i1> %mask)
  ret void
}

; Pointers are gathered as often as integers are, and the element size that
; decides how many lanes a part holds has to come from the pointer's own width
; rather than from an integer type. Eight pointer-sized lanes fill one AVX-512
; part and two AVX2 parts.
; SKX-COST-LABEL: 'gather_v8ptr_qword_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v8ptr_qword_index'
; SKX-TPUT: cost of 10 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8ptr_qword_index:
; SKX-ASM-COUNT-1: vpgatherqq
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v8ptr_qword_index'
; AVX2-COST: cost of 6 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v8ptr_qword_index:
; AVX2-ASM-COUNT-2: vpgatherqq
; AVX2-ASM-NOT: vpgatherqq
define <8 x ptr> @gather_v8ptr_qword_index(ptr %base, <8 x i64> %idx, <8 x i1> %mask) {
  %ptrs = getelementptr inbounds ptr, ptr %base, <8 x i64> %idx
  %r = call <8 x ptr> @llvm.masked.gather.v8p0.v8p0(<8 x ptr> %ptrs, i32 8, <8 x i1> %mask, <8 x ptr> poison)
  ret <8 x ptr> %r
}

; The same gather with only its first two lanes left live by a compile-time
; mask. The second AVX2 part holds no live lane and is not emitted, and both
; targets charge two lanes rather than eight.
; SKX-COST-LABEL: 'gather_v8ptr_qword_index_first2_mask'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v8ptr_qword_index_first2_mask'
; SKX-TPUT: cost of 4 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8ptr_qword_index_first2_mask:
; SKX-ASM-COUNT-1: vpgatherqq
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v8ptr_qword_index_first2_mask'
; AVX2-COST: cost of 1 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v8ptr_qword_index_first2_mask:
; AVX2-ASM-COUNT-1: vpgatherqq
; AVX2-ASM-NOT: vpgatherqq
define <8 x ptr> @gather_v8ptr_qword_index_first2_mask(ptr %base, <8 x i64> %idx) {
  %ptrs = getelementptr inbounds ptr, ptr %base, <8 x i64> %idx
  %r = call <8 x ptr> @llvm.masked.gather.v8p0.v8p0(<8 x ptr> %ptrs, i32 8, <8 x i1> <i1 1, i1 1, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0>, <8 x ptr> poison)
  ret <8 x ptr> %r
}

; A stride the scale field cannot encode -- it encodes 1, 2, 4 and 8 -- is
; multiplied into the index instead, and that product is what CodeGen narrows
; from. This GEP walks an array of three-word structures, the form a loop
; vectorizer emits for a[idx[i]].f, so its stride is twelve: a dword index
; scaled by twelve needs thirty-six bits and no longer fits a signed dword, so
; the gather ends up indexed by qwords, taking twice the instructions of the
; four-byte-stride control below.
; SKX-COST-LABEL: 'gather_v16i32_struct_stride'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_struct_stride:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_struct_stride'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_struct_stride:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgatherqd
define <16 x i32> @gather_v16i32_struct_stride(ptr %base, <16 x i32> %idx, <16 x i1> %mask) {
  %sext = sext <16 x i32> %idx to <16 x i64>
  %ptrs = getelementptr inbounds {i32, i32, i32}, ptr %base, <16 x i64> %sext, i32 0
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; The same stride over an index narrow enough that the product still fits a
; signed dword, so a nonencodable stride does not by itself decide the width.
; Fitting is not sufficient either: the multiply leaves the index as something
; other than an extension node, and CodeGen then narrows it only where that
; removes an illegal type. Sixteen qword indices are illegal everywhere, but
; the dword form they would narrow to is itself legal only once a 512-bit
; register is available, so the same gather keeps qword indices on AVX2 and
; narrows to dwords on AVX-512.
; SKX-COST-LABEL: 'gather_v16i32_struct_stride_narrow_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_struct_stride_narrow_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_struct_stride_narrow_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_struct_stride_narrow_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgatherqd
; KNL-COST-LABEL: 'gather_v16i32_struct_stride_narrow_index'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_struct_stride_narrow_index:
; KNL-ASM-COUNT-1: vpgatherdd
; KNL-ASM-NOT: vpgather
; KNL-ASM: retq
define <16 x i32> @gather_v16i32_struct_stride_narrow_index(ptr %base, <16 x i16> %idx, <16 x i1> %mask) {
  %sext = sext <16 x i16> %idx to <16 x i64>
  %ptrs = getelementptr inbounds {i32, i32, i32}, ptr %base, <16 x i64> %sext, i32 0
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; The same stride at the boundary the combine draws. ComputeNumSignBits adds
; the significant widths of the two multiply operands, and the constant 12
; spends one of its five bits on the sign, so a 28-bit index needs 33 and stays
; a qword. Measuring the stride by its magnitude alone leaves it at four bits,
; which admits this index and halves the count.
; SKX-COST-LABEL: 'gather_v16i32_struct_stride_28bit_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_struct_stride_28bit_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_struct_stride_28bit_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_struct_stride_28bit_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_struct_stride_28bit_index(ptr %base, <16 x i64> %idx, <16 x i1> %mask) {
  %and = and <16 x i64> %idx, splat (i64 134217727)
  %ptrs = getelementptr inbounds {i32, i32, i32}, ptr %base, <16 x i64> %and, i32 0
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; One bit narrower, which is the control: the product needs exactly 32 and the
; index narrows, so the two cases have to be priced differently.
; SKX-COST-LABEL: 'gather_v16i32_struct_stride_27bit_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_struct_stride_27bit_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_struct_stride_27bit_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_struct_stride_27bit_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_struct_stride_27bit_index(ptr %base, <16 x i64> %idx, <16 x i1> %mask) {
  %and = and <16 x i64> %idx, splat (i64 67108863)
  %ptrs = getelementptr inbounds {i32, i32, i32}, ptr %base, <16 x i64> %and, i32 0
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; The same gather over a four-byte stride, which the scale field does encode,
; so the index stays a dword and one instruction covers twice the lanes.
; SKX-COST-LABEL: 'gather_v16i32_scaled_stride'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_scaled_stride:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_scaled_stride'
; AVX2-COST: cost of 6 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_scaled_stride:
; AVX2-ASM-COUNT-2: vpgatherdd
; AVX2-ASM-NOT: vpgatherdd
define <16 x i32> @gather_v16i32_scaled_stride(ptr %base, <16 x i32> %idx, <16 x i1> %mask) {
  %sext = sext <16 x i32> %idx to <16 x i64>
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %sext
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; An index that is neither a constant nor an extension has only the truncation
; route, which the combine takes only where it removes an illegal type. Sixteen
; qword indices are illegal everywhere, but the dword form they narrow to needs
; a 512-bit register, so the same gather keeps qword indices on AVX2.
; SKX-COST-LABEL: 'gather_v16i32_masked_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_masked_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_masked_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_masked_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_masked_index'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_masked_index:
; KNL-ASM-COUNT-1: vpgatherdd
; KNL-ASM-NOT: vpgather
; KNL-ASM: retq
define <16 x i32> @gather_v16i32_masked_index(ptr %base, <16 x i64> %x, <16 x i1> %mask) {
  %idx = and <16 x i64> %x, splat (i64 255)
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %idx
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; A zero-extension from a narrow type cannot reach the signed dword range, so
; the index stays a dword and one instruction covers all sixteen lanes, exactly
; as for the sign-extension above. Reading the extension opcode rather than the
; range it produces would price this as the qword form it is not.
; SKX-COST-LABEL: 'gather_v16i32_narrow_zext_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_narrow_zext_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_narrow_zext_index'
; AVX2-COST: cost of 6 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_narrow_zext_index:
; AVX2-ASM-COUNT-2: vpgatherdd
; AVX2-ASM-NOT: vpgatherdd
define <16 x i32> @gather_v16i32_narrow_zext_index(ptr %base, <16 x i8> %idx, <16 x i1> %mask) {
  %zext = zext <16 x i8> %idx to <16 x i64>
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %zext
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; Zero-extending a dword is the case that has to stay distinct from it. The top
; bit is no longer a sign, so the value can exceed what a signed dword index
; addresses and the index stays a qword, taking twice the instructions of the
; sign-extended form.
; SKX-COST-LABEL: 'gather_v16i32_wide_zext_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_wide_zext_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_wide_zext_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_wide_zext_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgatherqd
define <16 x i32> @gather_v16i32_wide_zext_index(ptr %base, <16 x i32> %idx, <16 x i1> %mask) {
  %zext = zext <16 x i32> %idx to <16 x i64>
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %zext
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; A constant index that does fit a signed dword stays one, the control for the
; out-of-range form below.
; SKX-COST-LABEL: 'gather_v8i32_const_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8i32_const_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v8i32_const_index'
; AVX2-COST: cost of 1 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v8i32_const_index:
; AVX2-ASM-COUNT-1: vpgatherdd
; AVX2-ASM-NOT: vpgatherdd
define <8 x i32> @gather_v8i32_const_index(ptr %base, <8 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <8 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

; A constant index is an index like any other, and it has to be examined like
; one: these values do not fit a signed dword, so the gather ends up indexed by
; qwords. Eight qword indices fill one AVX-512 register, which emits a single
; instruction at either width, so only AVX2 shows the split.
; SKX-COST-LABEL: 'gather_v8i32_const_index_out_of_range'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8i32_const_index_out_of_range:
; SKX-ASM-COUNT-1: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v8i32_const_index_out_of_range'
; AVX2-COST: cost of 2 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v8i32_const_index_out_of_range:
; AVX2-ASM-COUNT-2: vpgatherqd
; AVX2-ASM-NOT: vpgatherqd
define <8 x i32> @gather_v8i32_const_index_out_of_range(ptr %base, <8 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <8 x i64> <i64 0, i64 2147483648, i64 4294967296, i64 6442450944, i64 8589934592, i64 10737418240, i64 12884901888, i64 15032385536>
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

; A constant splat added to the index is scaled and folded into the base, so
; what reaches the instruction is the extension underneath it. Reading the add
; instead finds 41 significant bits and prices a qword form CodeGen never
; emits.
; SKX-COST-LABEL: 'gather_v16i32_splat_addend'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_splat_addend:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_splat_addend'
; AVX2-COST: cost of 6 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_splat_addend:
; AVX2-ASM-COUNT-2: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_splat_addend(ptr %base, <16 x i16> %idx, <16 x i1> %mask) {
  %sext = sext <16 x i16> %idx to <16 x i64>
  %add = add <16 x i64> %sext, splat (i64 1099511627776)
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %add
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; Every index is a constant splat, so nothing in the index list varies. A
; vector GEP spells out every lane, so a uniform base leaves the whole address
; uniform and the index folds to a dword. Treating no varying index as proof
; that the base varies would price this as a vector of pointers.
; SKX-COST-LABEL: 'gather_v8i32_splat_const_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8i32_splat_const_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v8i32_splat_const_index'
; AVX2-COST: cost of 2 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v8i32_splat_const_index:
; AVX2-ASM-COUNT-1: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <8 x i32> @gather_v8i32_splat_const_index(ptr %base, <8 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <8 x i64> zeroinitializer
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

; A GEP whose scalar index is uniform but not a constant. It contributes a fixed
; byte offset that folds into the base pointer, so it does not widen the index
; operand and the operation stays one part.
; SKX-COST-LABEL: 'gather_v16i32_uniform_scalar_gep_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_uniform_scalar_gep_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_uniform_scalar_gep_index'
; AVX2-COST: cost of 4 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_uniform_scalar_gep_index:
; AVX2-ASM-COUNT-2: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_uniform_scalar_gep_index(ptr %base, i64 %k, <16 x i32> %idx) {
  %ptrs = getelementptr inbounds [8 x i32], ptr %base, i64 %k, <16 x i32> %idx
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> splat (i1 true), <16 x i32> poison)
  ret <16 x i32> %v
}

; A narrow index whose narrowness only a PHI carries, reached through a shift.
; ValueTracking walks the PHI; SelectionDAG stops at the block boundary, so the
; index stays qword and the gather splits.
; SKX-COST-LABEL: 'gather_v16i32_phi_through_shift'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_phi_through_shift:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_phi_through_shift'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_phi_through_shift:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_phi_through_shift'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_phi_through_shift:
; KNL-ASM-COUNT-2: vpgatherqd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_phi_through_shift(ptr %base, i1 %c, <16 x i1> %mask) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %join
f:
  br label %join
join:
  %p = phi <16 x i64> [ splat (i64 7), %t ], [ splat (i64 9), %f ]
  %narrow = shl <16 x i64> %p, splat (i64 2)
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %narrow
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; The same for a range attribute, which lives in the IR and not in the DAG.
; SKX-COST-LABEL: 'gather_v16i32_range_through_shift'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_range_through_shift:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_range_through_shift'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_range_through_shift:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_range_through_shift'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_range_through_shift:
; KNL-ASM-COUNT-2: vpgatherqd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_range_through_shift(ptr %base, <16 x i64> range(i64 0, 1024) %idx, <16 x i1> %mask) {
  %narrow = shl <16 x i64> %idx, splat (i64 2)
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %narrow
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, i32, <16 x i1>, <16 x i32>)
declare <8 x ptr> @llvm.masked.gather.v8p0.v8p0(<8 x ptr>, i32, <8 x i1>, <8 x ptr>)
declare <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr>, i32, <8 x i1>, <8 x i32>)
declare <9 x i32> @llvm.masked.gather.v9i32.v9p0(<9 x ptr>, i32, <9 x i1>, <9 x i32>)
declare <17 x i32> @llvm.masked.gather.v17i32.v17p0(<17 x ptr>, i32, <17 x i1>, <17 x i32>)
declare <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr>, i32, <24 x i1>, <24 x i32>)
declare void @llvm.masked.scatter.v9i32.v9p0(<9 x i32>, <9 x ptr>, i32, <9 x i1>)
declare void @llvm.masked.scatter.v24i32.v24p0(<24 x i32>, <24 x ptr>, i32, <24 x i1>)
declare void @llvm.masked.scatter.v48i32.v48p0(<48 x i32>, <48 x ptr>, i32, <48 x i1>)

attributes #0 = { "prefer-vector-width"="256" "min-legal-vector-width"="0" }

declare void @consume_ptrs(<8 x ptr>)

; A range attribute bounds the index to ten bits, but SelectionDAG does not
; carry an argument's attribute, so its own sign-bit analysis sees a full qword
; and keeps the wider index. Narrowing on a fact only the IR holds halves the
; count.
; SKX-COST-LABEL: 'gather_v16i32_range_attr_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_range_attr_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_range_attr_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_range_attr_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_range_attr_index'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_range_attr_index:
; KNL-ASM-COUNT-2: vpgatherqd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_range_attr_index(ptr %base, <16 x i64> range(i64 0, 1024) %idx, <16 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %idx
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; Both incoming values are narrow, but the PHI carries them across a block
; boundary, where SelectionDAG's analysis does not follow. The index stays a
; qword.
; SKX-COST-LABEL: 'gather_v16i32_phi_index'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_phi_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_phi_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_phi_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_phi_index'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_phi_index:
; KNL-ASM-COUNT-2: vpgatherqd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_phi_index(ptr %base, <16 x i16> %a, <16 x i16> %b, i1 %c, <16 x i1> %mask) {
entry:
  br i1 %c, label %left, label %right
left:
  %ea = sext <16 x i16> %a to <16 x i64>
  br label %join
right:
  %eb = sext <16 x i16> %b to <16 x i64>
  br label %join
join:
  %idx = phi <16 x i64> [ %ea, %left ], [ %eb, %right ]
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %idx
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; A constant splat added to the index folds into the base only where the stride
; leaves it separable. A stride of twelve multiplies the whole sum, so the
; product the combine tests includes the addend and overflows a signed dword.
; Peeling the addend regardless admits the narrow index the other addend has.
; SKX-COST-LABEL: 'gather_v16i32_struct_stride_big_addend'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_struct_stride_big_addend:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_struct_stride_big_addend'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_struct_stride_big_addend:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_struct_stride_big_addend'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_struct_stride_big_addend:
; KNL-ASM-COUNT-2: vpgatherqd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_struct_stride_big_addend(ptr %base, <16 x i16> %idx, <16 x i1> %mask) {
  %wide = sext <16 x i16> %idx to <16 x i64>
  %offset = add <16 x i64> %wide, splat (i64 2147483648)
  %ptrs = getelementptr {i32, i32, i32}, ptr %base, <16 x i64> %offset, i32 0
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; The same addend under a power-of-two stride, where the multiply becomes a
; shift the constant term survives. Here the addend does fold away and the index
; narrows, so the peel must not be dropped altogether.
; SKX-COST-LABEL: 'gather_v16i32_pow2_stride_big_addend'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_pow2_stride_big_addend:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_pow2_stride_big_addend'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_pow2_stride_big_addend:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_pow2_stride_big_addend'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_pow2_stride_big_addend:
; KNL-ASM-COUNT-1: vpgatherdd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_pow2_stride_big_addend(ptr %base, <16 x i16> %idx, <16 x i1> %mask) {
  %wide = sext <16 x i16> %idx to <16 x i64>
  %offset = add <16 x i64> %wide, splat (i64 2147483648)
  %ptrs = getelementptr {i32, i32, i32, i32}, ptr %base, <16 x i64> %offset, i32 0
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; getUniformBase() declines a GEP from a block other than the gather's, so the
; address arrives as a materialized pointer vector with no arithmetic left to
; narrow. The same GEP in the gather's own block does narrow, which the next
; function checks.
; SKX-COST-LABEL: 'gather_v8i32_gep_other_block'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8i32_gep_other_block:
; SKX-ASM-COUNT-1: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v8i32_gep_other_block'
; AVX2-COST: cost of 1 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v8i32_gep_other_block:
; AVX2-ASM-COUNT-2: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v8i32_gep_other_block'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v8i32_gep_other_block:
; KNL-ASM-COUNT-1: vpgatherqd
; KNL-ASM-NOT: vpgather
define <8 x i32> @gather_v8i32_gep_other_block(ptr %base, <8 x i32> %idx, <8 x i1> %mask) {
entry:
  %ptrs = getelementptr i32, ptr %base, <8 x i32> %idx
  call void @consume_ptrs(<8 x ptr> %ptrs)
  br label %use
use:
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

; The control for the block check.
; SKX-COST-LABEL: 'gather_v8i32_gep_same_block'
; SKX-COST: cost of 1 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v8i32_gep_same_block:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v8i32_gep_same_block'
; AVX2-COST: cost of 1 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v8i32_gep_same_block:
; AVX2-ASM-COUNT-1: vpgatherdd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v8i32_gep_same_block'
; KNL-COST: cost of 1 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v8i32_gep_same_block:
; KNL-ASM-COUNT-1: vpgatherdd
; KNL-ASM-NOT: vpgather
define <8 x i32> @gather_v8i32_gep_same_block(ptr %base, <8 x i32> %idx, <8 x i1> %mask) {
  %ptrs = getelementptr i32, ptr %base, <8 x i32> %idx
  call void @consume_ptrs(<8 x ptr> %ptrs)
  %v = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> poison)
  ret <8 x i32> %v
}

; A stride of three is multiplied into the index, and the product still
; fits a signed dword here, so CodeGen narrows and emits one instruction.
; The index and the stride share one sign bit between them, not one each.
; SKX-COST-LABEL: 'gather_v16i32_stride3_narrow_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_stride3_narrow_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_stride3_narrow_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_stride3_narrow_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_stride3_narrow_index'
; KNL-COST: cost of 2 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_stride3_narrow_index:
; KNL-ASM-COUNT-1: vpgatherdd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_stride3_narrow_index(ptr %base, <16 x i64> %idx, <16 x i1> %mask) {
  %narrow = and <16 x i64> %idx, splat (i64 536870911)
  %ptrs = getelementptr inbounds [3 x i8], ptr %base, <16 x i64> %narrow
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; One bit wider than the shape above, so the product no longer fits and the
; index stays a qword. This is the control for it.
; SKX-COST-LABEL: 'gather_v16i32_stride3_wide_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_stride3_wide_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_stride3_wide_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_stride3_wide_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_stride3_wide_index'
; KNL-COST: cost of 2 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_stride3_wide_index:
; KNL-ASM-COUNT-2: vpgatherqd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_stride3_wide_index(ptr %base, <16 x i64> %idx, <16 x i1> %mask) {
  %narrow = and <16 x i64> %idx, splat (i64 1073741823)
  %ptrs = getelementptr inbounds [3 x i8], ptr %base, <16 x i64> %narrow
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; A power-of-two stride arrives as a shift, and combineGatherScatter walks
; it into the scale field until the field is full. A stride of sixteen
; leaves one bit of shift behind, not four, so this still narrows.
; SKX-COST-LABEL: 'gather_v16i32_stride16_narrow_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_stride16_narrow_index:
; SKX-ASM-COUNT-1: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_stride16_narrow_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_stride16_narrow_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_stride16_narrow_index'
; KNL-COST: cost of 2 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_stride16_narrow_index:
; KNL-ASM-COUNT-1: vpgatherdd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_stride16_narrow_index(ptr %base, <16 x i64> %idx, <16 x i1> %mask) {
  %narrow = and <16 x i64> %idx, splat (i64 1073741823)
  %ptrs = getelementptr inbounds [16 x i8], ptr %base, <16 x i64> %narrow
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}

; One bit wider than the shape above, where even the residual shift does
; not fit. This is the control for it.
; SKX-COST-LABEL: 'gather_v16i32_stride16_wide_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_stride16_wide_index:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_stride16_wide_index'
; AVX2-COST: cost of 8 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_stride16_wide_index:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgather
; KNL-COST-LABEL: 'gather_v16i32_stride16_wide_index'
; KNL-COST: cost of 2 for instruction: {{.*}}masked.gather
; KNL-ASM-LABEL: gather_v16i32_stride16_wide_index:
; KNL-ASM-COUNT-2: vpgatherqd
; KNL-ASM-NOT: vpgather
define <16 x i32> @gather_v16i32_stride16_wide_index(ptr %base, <16 x i64> %idx, <16 x i1> %mask) {
  %narrow = and <16 x i64> %idx, splat (i64 2147483647)
  %ptrs = getelementptr inbounds [16 x i8], ptr %base, <16 x i64> %narrow
  %v = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <16 x i32> poison)
  ret <16 x i32> %v
}
