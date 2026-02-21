; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 | %ptxas-verify %}

; Test !mem.cache_hint metadata on llvm.memcpy intrinsic
; For memcpy:
;   operand_no = 0 applies to destination (store side)
;   operand_no = 1 applies to source (load side)

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

;-----------------------------------------------------------------------------
; Test memcpy with cache hints on both source and destination
; Source (operand 1): L1::evict_first, L2::evict_first, L2::128B
; Dest (operand 0): L1::evict_last, L2::evict_last
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_memcpy_both_hints
; CHECK-DAG: ld.global.L1::evict_first.L2::evict_first.L2::128B.b
; CHECK-DAG: st.global.L1::evict_last.L2::evict_last.b
define void @test_memcpy_both_hints(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !80
  ret void
}

;-----------------------------------------------------------------------------
; Test memcpy with cache hint only on source (load side)
; Source (operand 1): L1::evict_first
; Dest (operand 0): no hint
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_memcpy_src_hint_only
; CHECK: ld.global.L1::evict_first.b
; CHECK: st.global.b
; CHECK-NOT: st.global.L1
define void @test_memcpy_src_hint_only(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !81
  ret void
}

;-----------------------------------------------------------------------------
; Test memcpy with cache hint only on destination (store side)
; Source (operand 1): no hint
; Dest (operand 0): L2::evict_last
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_memcpy_dest_hint_only
; CHECK: ld.global.b
; CHECK-NOT: ld.global.L2
; CHECK: st.global.L2::evict_last.b
define void @test_memcpy_dest_hint_only(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !82
  ret void
}

;-----------------------------------------------------------------------------
; Test memcpy with L2::cache_hint on both operands
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_memcpy_l2_cache_hint
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 34343
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 12121
; CHECK-DAG: ld.global.L2::cache_hint.b
; CHECK-DAG: st.global.L2::cache_hint.b
define void @test_memcpy_l2_cache_hint(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !83
  ret void
}

;-----------------------------------------------------------------------------
; Test memcpy without cache hints produces plain load/store
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_memcpy_no_hint
; CHECK: ld.global.b
; CHECK: st.global.b
; CHECK-NOT: L1::evict
; CHECK-NOT: L2::evict
; CHECK-NOT: L2::cache_hint
define void @test_memcpy_no_hint(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false)
  ret void
}

;-----------------------------------------------------------------------------
; Combined L1 + L2 eviction policies
;-----------------------------------------------------------------------------

; Source: L1::evict_first + L2::evict_last
; Dest: no hint
; CHECK-LABEL: test_memcpy_src_l1_l2_combined
; CHECK: ld.global.L1::evict_first.L2::evict_last.b
; CHECK: st.global.b
; CHECK-NOT: st.global.L1
; CHECK-NOT: st.global.L2
define void @test_memcpy_src_l1_l2_combined(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !84
  ret void
}

; Source: no hint
; Dest: L1::evict_unchanged + L2::evict_first
; CHECK-LABEL: test_memcpy_dest_l1_l2_combined
; CHECK: ld.global.b
; CHECK-NOT: ld.global.L1
; CHECK: st.global.L1::evict_unchanged.L2::evict_first.b
define void @test_memcpy_dest_l1_l2_combined(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !85
  ret void
}

;-----------------------------------------------------------------------------
; L1 + prefetch combinations
;-----------------------------------------------------------------------------

; Source: L1::evict_last + L2::256B prefetch
; Dest: L1::no_allocate
; CHECK-LABEL: test_memcpy_l1_prefetch
; CHECK-DAG: ld.global.L1::evict_last.L2::256B.b
; CHECK-DAG: st.global.L1::no_allocate.b
define void @test_memcpy_l1_prefetch(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !86
  ret void
}

; Source: L2::64B prefetch only
; Dest: L2::evict_last only
; CHECK-LABEL: test_memcpy_prefetch_vs_eviction
; CHECK: ld.global.L2::64B.b
; CHECK: st.global.L2::evict_last.b
define void @test_memcpy_prefetch_vs_eviction(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !87
  ret void
}

;-----------------------------------------------------------------------------
; L2::cache_hint combined with other hints
;-----------------------------------------------------------------------------

; Source: L2::cache_hint + L1::evict_first
; Dest: no hint
; CHECK-LABEL: test_memcpy_src_cache_hint_l1
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 55555
; CHECK: ld.global.L1::evict_first.L2::cache_hint.b
; CHECK: st.global.b
; CHECK-NOT: st.global.L2::cache_hint
define void @test_memcpy_src_cache_hint_l1(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !88
  ret void
}

; Source: no hint
; Dest: L2::cache_hint + L1::evict_last + L2::evict_first
; CHECK-LABEL: test_memcpy_dest_cache_hint_combined
; CHECK: ld.global.b
; CHECK-NOT: ld.global.L2::cache_hint
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 66666
; CHECK: st.global.L1::evict_last.L2::evict_first.L2::cache_hint.b
define void @test_memcpy_dest_cache_hint_combined(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !89
  ret void
}

; Both operands: L2::cache_hint + L1 eviction + L2 eviction
; Source: L2::cache_hint(77777) + L1::evict_unchanged + L2::evict_last
; Dest: L2::cache_hint(88888) + L1::evict_first + L2::evict_first
; CHECK-LABEL: test_memcpy_both_cache_hint_combined
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 77777
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 88888
; CHECK-DAG: ld.global.L1::evict_unchanged.L2::evict_last.L2::cache_hint.b
; CHECK-DAG: st.global.L1::evict_first.L2::evict_first.L2::cache_hint.b
define void @test_memcpy_both_cache_hint_combined(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !90
  ret void
}

;-----------------------------------------------------------------------------
; L2::cache_hint + prefetch combinations
;-----------------------------------------------------------------------------

; Source: L2::cache_hint + L2::128B prefetch
; Dest: L2::cache_hint + L1::evict_last
; CHECK-LABEL: test_memcpy_cache_hint_prefetch
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 11111
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 22222
; CHECK-DAG: ld.global.L2::cache_hint.L2::128B.b
; CHECK-DAG: st.global.L1::evict_last.L2::cache_hint.b
define void @test_memcpy_cache_hint_prefetch(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !91
  ret void
}

;-----------------------------------------------------------------------------
; Asymmetric hint combinations (complex vs simple)
;-----------------------------------------------------------------------------

; Source: all hints (L1 + L2 eviction + prefetch)
; Dest: simple L1 hint only
; CHECK-LABEL: test_memcpy_complex_src_simple_dest
; CHECK-DAG: ld.global.L1::evict_first.L2::evict_last.L2::64B.b
; CHECK-DAG: st.global.L1::evict_last.b
define void @test_memcpy_complex_src_simple_dest(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !92
  ret void
}

; Source: simple L2 prefetch only
; Dest: all hints (L1 + L2 eviction + L2::cache_hint)
; CHECK-LABEL: test_memcpy_simple_src_complex_dest
; CHECK: ld.global.L2::256B.b
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 99999
; CHECK: st.global.L1::no_allocate.L2::evict_last.L2::cache_hint.b
define void @test_memcpy_simple_src_complex_dest(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !93
  ret void
}

;-----------------------------------------------------------------------------
; Different L1 eviction policies on src vs dest
;-----------------------------------------------------------------------------

; Source: L1::evict_unchanged
; Dest: L1::evict_first
; CHECK-LABEL: test_memcpy_different_l1_policies
; CHECK-DAG: ld.global.L1::evict_unchanged.b
; CHECK-DAG: st.global.L1::evict_first.b
define void @test_memcpy_different_l1_policies(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !94
  ret void
}

; Source: L1::no_allocate
; Dest: L1::evict_unchanged
; CHECK-LABEL: test_memcpy_no_allocate_vs_unchanged
; CHECK-DAG: ld.global.L1::no_allocate.b
; CHECK-DAG: st.global.L1::evict_unchanged.b
define void @test_memcpy_no_allocate_vs_unchanged(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !95
  ret void
}

;-----------------------------------------------------------------------------
; All hints maxed out on both operands
;-----------------------------------------------------------------------------

; Source: L1::evict_first + L2::evict_first + L2::256B + L2::cache_hint
; Dest: L1::evict_last + L2::evict_last + L2::128B + L2::cache_hint
; CHECK-LABEL: test_memcpy_all_hints_both
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 12345
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 67890
; CHECK-DAG: ld.global.L1::evict_first.L2::evict_first.L2::cache_hint.L2::256B.b
; CHECK-DAG: st.global.L1::evict_last.L2::evict_last.L2::cache_hint.L2::128B.b
define void @test_memcpy_all_hints_both(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 4, i1 false), !mem.cache_hint !96
  ret void
}

;-----------------------------------------------------------------------------
; Metadata definitions
;-----------------------------------------------------------------------------

; memcpy with both dest and src hints
!80 = !{i32 0, !180, i32 1, !181}
; operand 0 (dest/store): L1::evict_last, L2::evict_last
!180 = !{!"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"last"}
; operand 1 (src/load): L1::evict_first, L2::evict_first, L2::128B prefetch
!181 = !{!"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"first", !"nvvm.l2_prefetch_size", !"128B"}

; memcpy with only source hint (load side)
!81 = !{i32 1, !182}
!182 = !{!"nvvm.l1_eviction", !"first"}

; memcpy with only dest hint (store side)
!82 = !{i32 0, !183}
!183 = !{!"nvvm.l2_eviction", !"last"}

; memcpy with L2::cache_hint on both operands
!83 = !{i32 0, !184, i32 1, !185}
!184 = !{!"nvvm.l2_cache_hint", i64 12121}
!185 = !{!"nvvm.l2_cache_hint", i64 34343}

; Combined L1 + L2 eviction on source only
!84 = !{i32 1, !186}
!186 = !{!"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"last"}

; Combined L1 + L2 eviction on dest only
!85 = !{i32 0, !187}
!187 = !{!"nvvm.l1_eviction", !"unchanged", !"nvvm.l2_eviction", !"first"}

; L1 + prefetch on source, L1 on dest
!86 = !{i32 1, !188, i32 0, !189}
!188 = !{!"nvvm.l1_eviction", !"last", !"nvvm.l2_prefetch_size", !"256B"}
!189 = !{!"nvvm.l1_eviction", !"no_allocate"}

; Prefetch on source, L2 eviction on dest
!87 = !{i32 1, !190, i32 0, !191}
!190 = !{!"nvvm.l2_prefetch_size", !"64B"}
!191 = !{!"nvvm.l2_eviction", !"last"}

; L2::cache_hint + L1 eviction on source only
!88 = !{i32 1, !192}
!192 = !{!"nvvm.l2_cache_hint", i64 55555, !"nvvm.l1_eviction", !"first"}

; L2::cache_hint + L1 + L2 eviction on dest only
!89 = !{i32 0, !193}
!193 = !{!"nvvm.l2_cache_hint", i64 66666, !"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"first"}

; Both operands: L2::cache_hint + L1 + L2 eviction
!90 = !{i32 1, !194, i32 0, !195}
!194 = !{!"nvvm.l2_cache_hint", i64 77777, !"nvvm.l1_eviction", !"unchanged", !"nvvm.l2_eviction", !"last"}
!195 = !{!"nvvm.l2_cache_hint", i64 88888, !"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"first"}

; L2::cache_hint + prefetch on source, L2::cache_hint + L1 on dest
!91 = !{i32 1, !196, i32 0, !197}
!196 = !{!"nvvm.l2_cache_hint", i64 11111, !"nvvm.l2_prefetch_size", !"128B"}
!197 = !{!"nvvm.l2_cache_hint", i64 22222, !"nvvm.l1_eviction", !"last"}

; Complex source (all non-cache_hint), simple dest
!92 = !{i32 1, !198, i32 0, !199}
!198 = !{!"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"last", !"nvvm.l2_prefetch_size", !"64B"}
!199 = !{!"nvvm.l1_eviction", !"last"}

; Simple source, complex dest (with cache_hint)
!93 = !{i32 1, !200, i32 0, !201}
!200 = !{!"nvvm.l2_prefetch_size", !"256B"}
!201 = !{!"nvvm.l2_cache_hint", i64 99999, !"nvvm.l1_eviction", !"no_allocate", !"nvvm.l2_eviction", !"last"}

; Different L1 policies: unchanged vs first
!94 = !{i32 1, !202, i32 0, !203}
!202 = !{!"nvvm.l1_eviction", !"unchanged"}
!203 = !{!"nvvm.l1_eviction", !"first"}

; Different L1 policies: no_allocate vs unchanged
!95 = !{i32 1, !204, i32 0, !205}
!204 = !{!"nvvm.l1_eviction", !"no_allocate"}
!205 = !{!"nvvm.l1_eviction", !"unchanged"}

; All hints maxed out on both operands
!96 = !{i32 1, !206, i32 0, !207}
!206 = !{!"nvvm.l2_cache_hint", i64 12345, !"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"first", !"nvvm.l2_prefetch_size", !"256B"}
!207 = !{!"nvvm.l2_cache_hint", i64 67890, !"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"last", !"nvvm.l2_prefetch_size", !"128B"}

;-----------------------------------------------------------------------------
; Large memcpy tests - verify hints propagate to all expanded load/stores
; LLVM expands memcpy to multiple load/store pairs. Each pair should
; get the appropriate cache hints from the original memcpy metadata.
; The expansion may use various sizes (b8, b16, b32, v2, v4, etc.)
;-----------------------------------------------------------------------------

; 16-byte memcpy: verify hints are applied to expanded loads/stores
; CHECK-LABEL: test_memcpy_16bytes
; CHECK: ld.global.L1::evict_first.b
; CHECK: st.global.L1::evict_last.b
define void @test_memcpy_16bytes(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 16, i1 false), !mem.cache_hint !97
  ret void
}

; 32-byte memcpy: all loads should have L1::evict_unchanged, all stores L2::evict_first
; CHECK-LABEL: test_memcpy_32bytes
; CHECK: ld.global.L1::evict_unchanged.b
; CHECK: st.global.L2::evict_first.b
define void @test_memcpy_32bytes(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 32, i1 false), !mem.cache_hint !98
  ret void
}

; 64-byte memcpy with L2::cache_hint
; All loads and stores should get the L2::cache_hint with their respective policies
; CHECK-LABEL: test_memcpy_64bytes_cache_hint
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 11111
; CHECK-DAG: mov.b64 {{%rd[0-9]+}}, 22222
; CHECK: ld.global.L2::cache_hint.b
; CHECK: st.global.L2::cache_hint.b
define void @test_memcpy_64bytes_cache_hint(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 64, i1 false), !mem.cache_hint !99
  ret void
}

; 128-byte memcpy with combined hints
; Note: Large memcpy (>64 bytes) is expanded to a loop in the backend.
; Cache hints are not preserved for loop-based expansion.
; This test verifies the code compiles correctly.
; CHECK-LABEL: test_memcpy_128bytes_combined
; CHECK: ld.global.b
; CHECK: st.global.b
define void @test_memcpy_128bytes_combined(ptr addrspace(1) %dest, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dest, ptr addrspace(1) %src, i64 128, i1 false), !mem.cache_hint !100
  ret void
}

; Large memcpy metadata
!97 = !{i32 1, !208, i32 0, !209}
!208 = !{!"nvvm.l1_eviction", !"first"}
!209 = !{!"nvvm.l1_eviction", !"last"}

!98 = !{i32 1, !210, i32 0, !211}
!210 = !{!"nvvm.l1_eviction", !"unchanged"}
!211 = !{!"nvvm.l2_eviction", !"first"}

!99 = !{i32 1, !212, i32 0, !213}
!212 = !{!"nvvm.l2_cache_hint", i64 11111}
!213 = !{!"nvvm.l2_cache_hint", i64 22222}

!100 = !{i32 1, !214, i32 0, !215}
!214 = !{!"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"last", !"nvvm.l2_prefetch_size", !"256B"}
!215 = !{!"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"first"}
