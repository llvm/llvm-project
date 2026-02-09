; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_60 -mattr=+ptx74 | FileCheck %s --check-prefix=SM60
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx74 | FileCheck %s --check-prefix=SM70
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_75 -mattr=+ptx74 | FileCheck %s --check-prefix=SM75
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 | FileCheck %s --check-prefix=SM80
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s --check-prefix=SM80-PTX70
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_86 -mattr=+ptx74 | FileCheck %s --check-prefix=SM86
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | FileCheck %s --check-prefix=SM90

; Test SM version requirements for cache hints (from PTX ISA documentation):
; - L1::evict_* requires SM 70+
; - L2::evict_* requires SM 70+
; - L2::64B and L2::128B require SM 75+
; - L2::256B requires SM 80+
; - L2::cache_hint requires SM 80+ and PTX 7.4+

;-----------------------------------------------------------------------------
; L1 eviction - requires SM 70+
; SM60 should NOT emit L1::evict_first (fall back to plain load)
; SM70+ should emit L1::evict_first
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_l1_first
; SM60: ld.global.b32
; SM60-NOT: L1::evict_first

; SM70-LABEL: test_load_l1_first
; SM70: ld.global.L1::evict_first.b32

; SM75-LABEL: test_load_l1_first
; SM75: ld.global.L1::evict_first.b32

; SM80-LABEL: test_load_l1_first
; SM80: ld.global.L1::evict_first.b32

; SM86-LABEL: test_load_l1_first
; SM86: ld.global.L1::evict_first.b32

; SM90-LABEL: test_load_l1_first
; SM90: ld.global.L1::evict_first.b32
define i32 @test_load_l1_first(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2 eviction - requires SM 70+
; SM60 should NOT emit L2::evict_last (fall back to plain load)
; SM70+ should emit L2::evict_last
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_l2_last
; SM60: ld.global.b32
; SM60-NOT: L2::evict_last

; SM70-LABEL: test_load_l2_last
; SM70: ld.global.L2::evict_last.b32

; SM75-LABEL: test_load_l2_last
; SM75: ld.global.L2::evict_last.b32

; SM80-LABEL: test_load_l2_last
; SM80: ld.global.L2::evict_last.b32

; SM86-LABEL: test_load_l2_last
; SM86: ld.global.L2::evict_last.b32

; SM90-LABEL: test_load_l2_last
; SM90: ld.global.L2::evict_last.b32
define i32 @test_load_l2_last(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !1
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2::64B prefetch - requires SM 75+
; SM60/SM70 should NOT emit L2::64B (fall back to plain load)
; SM75+ should emit L2::64B
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_prefetch_64
; SM60: ld.global.b32
; SM60-NOT: L2::64B

; SM70-LABEL: test_load_prefetch_64
; SM70: ld.global.b32
; SM70-NOT: L2::64B

; SM75-LABEL: test_load_prefetch_64
; SM75: ld.global.L2::64B.b32

; SM80-LABEL: test_load_prefetch_64
; SM80: ld.global.L2::64B.b32

; SM86-LABEL: test_load_prefetch_64
; SM86: ld.global.L2::64B.b32

; SM90-LABEL: test_load_prefetch_64
; SM90: ld.global.L2::64B.b32
define i32 @test_load_prefetch_64(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !6
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2::128B prefetch - requires SM 75+
; SM60/SM70 should NOT emit L2::128B (fall back to plain load)
; SM75+ should emit L2::128B
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_prefetch_128
; SM60: ld.global.b32
; SM60-NOT: L2::128B

; SM70-LABEL: test_load_prefetch_128
; SM70: ld.global.b32
; SM70-NOT: L2::128B

; SM75-LABEL: test_load_prefetch_128
; SM75: ld.global.L2::128B.b32

; SM80-LABEL: test_load_prefetch_128
; SM80: ld.global.L2::128B.b32

; SM86-LABEL: test_load_prefetch_128
; SM86: ld.global.L2::128B.b32

; SM90-LABEL: test_load_prefetch_128
; SM90: ld.global.L2::128B.b32
define i32 @test_load_prefetch_128(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !2
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2::256B prefetch - requires SM 80+
; SM60/SM70/SM75 should NOT emit L2::256B (fall back to plain load)
; SM80+ should emit L2::256B
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_prefetch_256
; SM60: ld.global.b32
; SM60-NOT: L2::256B

; SM70-LABEL: test_load_prefetch_256
; SM70: ld.global.b32
; SM70-NOT: L2::256B

; SM75-LABEL: test_load_prefetch_256
; SM75: ld.global.b32
; SM75-NOT: L2::256B

; SM80-LABEL: test_load_prefetch_256
; SM80: ld.global.L2::256B.b32

; SM86-LABEL: test_load_prefetch_256
; SM86: ld.global.L2::256B.b32

; SM90-LABEL: test_load_prefetch_256
; SM90: ld.global.L2::256B.b32
define i32 @test_load_prefetch_256(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !7
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2::cache_hint - requires SM 80+ and PTX 7.4+
; SM60/SM70/SM75 should NOT emit L2::cache_hint (fall back to plain load)
; SM80 with PTX < 7.4 should NOT emit L2::cache_hint
; SM80+ with PTX 7.4+ should emit L2::cache_hint
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_cache_hint
; SM60: ld.global.b32
; SM60-NOT: L2::cache_hint

; SM70-LABEL: test_load_cache_hint
; SM70: ld.global.b32
; SM70-NOT: L2::cache_hint

; SM75-LABEL: test_load_cache_hint
; SM75: ld.global.b32
; SM75-NOT: L2::cache_hint

; SM80-LABEL: test_load_cache_hint
; SM80: mov.b64 [[POLICY:%rd[0-9]+]], 12345
; SM80: ld.global.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]

; SM80-PTX70-LABEL: test_load_cache_hint
; SM80-PTX70: ld.global.b32
; SM80-PTX70-NOT: L2::cache_hint

; SM86-LABEL: test_load_cache_hint
; SM86: mov.b64 [[POLICY:%rd[0-9]+]], 12345
; SM86: ld.global.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]

; SM90-LABEL: test_load_cache_hint
; SM90: mov.b64 [[POLICY:%rd[0-9]+]], 12345
; SM90: ld.global.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i32 @test_load_cache_hint(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !3
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2::cache_hint combined with L1 eviction on older SM
; Both hints should be dropped on SM60
; L1 hint emitted but L2::cache_hint dropped on SM70/SM75
; Both emitted on SM80+
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_cache_hint_with_l1
; SM60: ld.global.b32
; SM60-NOT: L1::evict_first
; SM60-NOT: L2::cache_hint

; SM70-LABEL: test_load_cache_hint_with_l1
; SM70: ld.global.L1::evict_first.b32
; SM70-NOT: L2::cache_hint

; SM75-LABEL: test_load_cache_hint_with_l1
; SM75: ld.global.L1::evict_first.b32
; SM75-NOT: L2::cache_hint

; SM80-LABEL: test_load_cache_hint_with_l1
; SM80: mov.b64 [[POLICY:%rd[0-9]+]], 44445
; SM80: ld.global.L1::evict_first.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]

; SM80-PTX70-LABEL: test_load_cache_hint_with_l1
; SM80-PTX70: ld.global.L1::evict_first.b32
; SM80-PTX70-NOT: L2::cache_hint
define i32 @test_load_cache_hint_with_l1(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !4
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2::128B combined with L1 eviction on older SM
; Both hints dropped on SM60
; L1 hint emitted but L2::128B dropped on SM70
; Both emitted on SM75+
;-----------------------------------------------------------------------------

; SM60-LABEL: test_load_prefetch_with_l1
; SM60: ld.global.b32
; SM60-NOT: L1::evict_first
; SM60-NOT: L2::128B

; SM70-LABEL: test_load_prefetch_with_l1
; SM70: ld.global.L1::evict_first.b32
; SM70-NOT: L2::128B

; SM75-LABEL: test_load_prefetch_with_l1
; SM75: ld.global.L1::evict_first.L2::128B.b32

; SM80-LABEL: test_load_prefetch_with_l1
; SM80: ld.global.L1::evict_first.L2::128B.b32
define i32 @test_load_prefetch_with_l1(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !8
  ret i32 %v
}

;-----------------------------------------------------------------------------
; Store with L2::cache_hint
;-----------------------------------------------------------------------------

; SM60-LABEL: test_store_cache_hint
; SM60: st.global.b32
; SM60-NOT: L2::cache_hint

; SM70-LABEL: test_store_cache_hint
; SM70: st.global.b32
; SM70-NOT: L2::cache_hint

; SM80-LABEL: test_store_cache_hint
; SM80: mov.b64 [[POLICY:%rd[0-9]+]], 67890
; SM80: st.global.L2::cache_hint.b32 [{{%rd[0-9]+}}], {{%r[0-9]+}}, [[POLICY]]

; SM80-PTX70-LABEL: test_store_cache_hint
; SM80-PTX70: st.global.b32
; SM80-PTX70-NOT: L2::cache_hint
define void @test_store_cache_hint(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !5
  ret void
}

;-----------------------------------------------------------------------------
; Store with L1 eviction hint
;-----------------------------------------------------------------------------

; SM60-LABEL: test_store_l1_no_allocate
; SM60: st.global.b32
; SM60-NOT: L1::no_allocate

; SM70-LABEL: test_store_l1_no_allocate
; SM70: st.global.L1::no_allocate.b32

; SM80-LABEL: test_store_l1_no_allocate
; SM80: st.global.L1::no_allocate.b32
define void @test_store_l1_no_allocate(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !9
  ret void
}

;-----------------------------------------------------------------------------
; Metadata definitions
;-----------------------------------------------------------------------------

; L1 eviction: first
!0 = !{i32 0, !100}
!100 = !{!"nvvm.l1_eviction", !"first"}

; L2 eviction: last
!1 = !{i32 0, !101}
!101 = !{!"nvvm.l2_eviction", !"last"}

; L2 prefetch: 128B
!2 = !{i32 0, !102}
!102 = !{!"nvvm.l2_prefetch_size", !"128B"}

; L2::cache_hint only
!3 = !{i32 0, !103}
!103 = !{!"nvvm.l2_cache_hint", i64 12345}

; L2::cache_hint + L1 eviction
!4 = !{i32 0, !104}
!104 = !{!"nvvm.l2_cache_hint", i64 44445, !"nvvm.l1_eviction", !"first"}

; L2::cache_hint for store
!5 = !{i32 0, !105}
!105 = !{!"nvvm.l2_cache_hint", i64 67890}

; L2 prefetch: 64B
!6 = !{i32 0, !106}
!106 = !{!"nvvm.l2_prefetch_size", !"64B"}

; L2 prefetch: 256B
!7 = !{i32 0, !107}
!107 = !{!"nvvm.l2_prefetch_size", !"256B"}

; L2 prefetch: 128B + L1 eviction
!8 = !{i32 0, !108}
!108 = !{!"nvvm.l2_prefetch_size", !"128B", !"nvvm.l1_eviction", !"first"}

; L1 eviction: no_allocate (for store)
!9 = !{i32 0, !109}
!109 = !{!"nvvm.l1_eviction", !"no_allocate"}
