; Pins the cost model's part count against the number of gather/scatter
; instructions CodeGen actually emits, for both index widths on an AVX-512 and
; an AVX2 target.
;
; The pairing is the point of the test: the code-size cost of a hardware
; gather/scatter is its part count, so every cost check below has a matching
; instruction count taken from the same module. Three properties would
; otherwise drift unnoticed. A part count taken from the legalized register
; count overstates the work when legalization widens to a power of two; an
; index width derived only for wide AVX-512 vectors leaves a dword-indexed
; operation tied with the qword-indexed form that CodeGen splits into more
; instructions; and the tail a scatter still emits under a zeroed mask reaches
; only as far as the predicate widens, which depends on AVX512BW.
;
; The throughput run pins the other half of the cost, the per-lane term, which
; code size does not expose: it is charged for the lanes that are really
; accessed, so it must follow the remainder lane of a length that does not fill
; its parts, and must drop the lanes a compile-time mask kills.
;
; These checks are maintained by hand rather than by
; update_analyze_test_checks.py, which does not know about the paired llc runs.

; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=code-size -disable-output -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefix=SKX-COST
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=throughput -disable-output -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefix=SKX-TPUT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 | FileCheck %s --check-prefix=SKX-ASM
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=code-size -disable-output -mcpu=skylake 2>&1 | FileCheck %s --check-prefix=AVX2-COST
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake | FileCheck %s --check-prefix=AVX2-ASM
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes="print<cost-model>" -cost-kind=code-size -disable-output -mcpu=knl 2>&1 | FileCheck %s --check-prefix=KNL-COST
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=knl | FileCheck %s --check-prefix=KNL-ASM

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
; AVX2-COST: cost of 2 for instruction: {{.*}}masked.gather
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
; AVX2-COST: cost of 3 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v9i32_qword_index:
; AVX2-ASM-COUNT-3: vpgatherqd
; AVX2-ASM-NOT: vpgather
define <9 x i32> @gather_v9i32_qword_index(ptr %base, <9 x i64> %idx, <9 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <9 x i64> %idx
  %v = call <9 x i32> @llvm.masked.gather.v9i32.v9p0(<9 x ptr> %ptrs, i32 4, <9 x i1> %mask, <9 x i32> poison)
  ret <9 x i32> %v
}

; A length below the width at which the index used to be examined, so both index
; widths would otherwise be priced the same on either target.
; SKX-COST-LABEL: 'gather_v17i32_dword_index'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v17i32_dword_index:
; SKX-ASM-COUNT-2: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v17i32_dword_index'
; AVX2-COST: cost of 3 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v17i32_dword_index:
; AVX2-ASM-COUNT-3: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <17 x i32> @gather_v17i32_dword_index(ptr %base, <17 x i32> %idx, <17 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <17 x i32> %idx
  %v = call <17 x i32> @llvm.masked.gather.v17i32.v17p0(<17 x ptr> %ptrs, i32 4, <17 x i1> %mask, <17 x i32> poison)
  ret <17 x i32> %v
}

; SKX-COST-LABEL: 'gather_v17i32_qword_index'
; SKX-COST: cost of 3 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v17i32_qword_index:
; SKX-ASM-COUNT-3: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v17i32_qword_index'
; AVX2-COST: cost of 5 for instruction: {{.*}}masked.gather
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
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_dword_index:
; SKX-ASM-COUNT-2: vpgatherdd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v24i32_dword_index'
; AVX2-COST: cost of 3 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v24i32_dword_index:
; AVX2-ASM-COUNT-3: vpgatherdd
; AVX2-ASM-NOT: vpgather
define <24 x i32> @gather_v24i32_dword_index(ptr %base, <24 x i32> %idx, <24 x i1> %mask) {
  %ptrs = getelementptr inbounds i32, ptr %base, <24 x i32> %idx
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> %mask, <24 x i32> poison)
  ret <24 x i32> %v
}

; SKX-COST-LABEL: 'gather_v24i32_qword_index'
; SKX-COST: cost of 3 for instruction: {{.*}}masked.gather
; SKX-TPUT-LABEL: 'gather_v24i32_qword_index'
; SKX-TPUT: cost of 30 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v24i32_qword_index:
; SKX-ASM-COUNT-3: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v24i32_qword_index'
; AVX2-COST: cost of 6 for instruction: {{.*}}masked.gather
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
; SKX-COST: cost of 4 for instruction: {{.*}}masked.scatter
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
; data. AVX512BW holds the mask in a k-register as wide as the legalized
; vector, so a 48-lane length rounds up to 64 and eight parts are emitted;
; without it the mask is broken into 16-lane pieces, the length rounds up only
; to 48, and six are. Counting legal registers would give eight on both.
; SKX-COST-LABEL: 'scatter_v48i32_qword_index_var_mask'
; SKX-COST: cost of 8 for instruction: {{.*}}masked.scatter
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
; SKX-COST: cost of 4 for instruction: {{.*}}masked.scatter
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
; AVX2-COST: cost of 2 for instruction: {{.*}}masked.gather
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

; A narrow index only stays narrow if the addressing mode can apply its stride
; as a scale, and the scale field encodes 1, 2, 4 and 8. This GEP walks an
; array of three-word structures, the form a loop vectorizer emits for
; a[idx[i]].f, so its stride is twelve: the multiply is folded into the index
; and the gather ends up indexed by qwords, taking twice the instructions of
; the four-byte-stride control below.
; SKX-COST-LABEL: 'gather_v16i32_struct_stride'
; SKX-COST: cost of 2 for instruction: {{.*}}masked.gather
; SKX-ASM-LABEL: gather_v16i32_struct_stride:
; SKX-ASM-COUNT-2: vpgatherqd
; SKX-ASM-NOT: vpgather
; AVX2-COST-LABEL: 'gather_v16i32_struct_stride'
; AVX2-COST: cost of 4 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_struct_stride:
; AVX2-ASM-COUNT-4: vpgatherqd
; AVX2-ASM-NOT: vpgatherqd
define <16 x i32> @gather_v16i32_struct_stride(ptr %base, <16 x i32> %idx, <16 x i1> %mask) {
  %sext = sext <16 x i32> %idx to <16 x i64>
  %ptrs = getelementptr inbounds {i32, i32, i32}, ptr %base, <16 x i64> %sext, i32 0
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
; AVX2-COST: cost of 2 for instruction: {{.*}}masked.gather
; AVX2-ASM-LABEL: gather_v16i32_scaled_stride:
; AVX2-ASM-COUNT-2: vpgatherdd
; AVX2-ASM-NOT: vpgatherdd
define <16 x i32> @gather_v16i32_scaled_stride(ptr %base, <16 x i32> %idx, <16 x i1> %mask) {
  %sext = sext <16 x i32> %idx to <16 x i64>
  %ptrs = getelementptr inbounds i32, ptr %base, <16 x i64> %sext
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
