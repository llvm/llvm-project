; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 | %ptxas-verify %}

; Test !mem.cache_hint metadata lowering to PTX cache qualifiers
; PTX supports the following cache qualifiers:
;   L1 eviction: L1::evict_first, L1::evict_last, L1::evict_unchanged, L1::no_allocate
;   L2 eviction: L2::evict_first, L2::evict_last
;   L2 prefetch: L2::64B, L2::128B, L2::256B

;-----------------------------------------------------------------------------
; Basic L1 eviction policies for loads
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_l1_first
; CHECK: ld.global.L1::evict_first.b32
define i32 @test_load_l1_first(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

; CHECK-LABEL: test_load_l1_last
; CHECK: ld.global.L1::evict_last.b32
define i32 @test_load_l1_last(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !1
  ret i32 %v
}

; CHECK-LABEL: test_load_l1_unchanged
; CHECK: ld.global.L1::evict_unchanged.b32
define i32 @test_load_l1_unchanged(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !2
  ret i32 %v
}

; CHECK-LABEL: test_load_l1_no_allocate
; CHECK: ld.global.L1::no_allocate.b32
define i32 @test_load_l1_no_allocate(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !3
  ret i32 %v
}

;-----------------------------------------------------------------------------
; Basic L2 eviction policies for loads
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_l2_first
; CHECK: ld.global.L2::evict_first.b32
define i32 @test_load_l2_first(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !4
  ret i32 %v
}

; CHECK-LABEL: test_load_l2_last
; CHECK: ld.global.L2::evict_last.b32
define i32 @test_load_l2_last(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !5
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L2 prefetch sizes for loads
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_prefetch_64
; CHECK: ld.global.L2::64B.b32
define i32 @test_load_prefetch_64(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !6
  ret i32 %v
}

; CHECK-LABEL: test_load_prefetch_128
; CHECK: ld.global.L2::128B.b32
define i32 @test_load_prefetch_128(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !7
  ret i32 %v
}

; CHECK-LABEL: test_load_prefetch_256
; CHECK: ld.global.L2::256B.b32
define i32 @test_load_prefetch_256(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !8
  ret i32 %v
}

;-----------------------------------------------------------------------------
; All L1 + L2 combinations for loads
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_l1_first_l2_first
; CHECK: ld.global.L1::evict_first.L2::evict_first.b32
define i32 @test_load_l1_first_l2_first(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !20
  ret i32 %v
}

; CHECK-LABEL: test_load_l1_first_l2_last
; CHECK: ld.global.L1::evict_first.L2::evict_last.b32
define i32 @test_load_l1_first_l2_last(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !21
  ret i32 %v
}

; CHECK-LABEL: test_load_l1_last_l2_first
; CHECK: ld.global.L1::evict_last.L2::evict_first.b32
define i32 @test_load_l1_last_l2_first(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !22
  ret i32 %v
}

; CHECK-LABEL: test_load_l1_last_l2_last
; CHECK: ld.global.L1::evict_last.L2::evict_last.b32
define i32 @test_load_l1_last_l2_last(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !23
  ret i32 %v
}

;-----------------------------------------------------------------------------
; L1 + L2 + Prefetch combination for loads
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_l1_first_l2_last_prefetch_128
; CHECK: ld.global.L1::evict_first.L2::evict_last.L2::128B.b32
define i32 @test_load_l1_first_l2_last_prefetch_128(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !24
  ret i32 %v
}

;-----------------------------------------------------------------------------
; Basic L1 eviction policies for stores
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_store_l1_first
; CHECK: st.global.L1::evict_first.b32
define void @test_store_l1_first(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !0
  ret void
}

; CHECK-LABEL: test_store_l1_last
; CHECK: st.global.L1::evict_last.b32
define void @test_store_l1_last(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !1
  ret void
}

; CHECK-LABEL: test_store_l1_unchanged
; CHECK: st.global.L1::evict_unchanged.b32
define void @test_store_l1_unchanged(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !2
  ret void
}

; CHECK-LABEL: test_store_l1_no_allocate
; CHECK: st.global.L1::no_allocate.b32
define void @test_store_l1_no_allocate(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !3
  ret void
}

;-----------------------------------------------------------------------------
; Basic L2 eviction policies for stores
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_store_l2_first
; CHECK: st.global.L2::evict_first.b32
define void @test_store_l2_first(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !4
  ret void
}

; CHECK-LABEL: test_store_l2_last
; CHECK: st.global.L2::evict_last.b32
define void @test_store_l2_last(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !5
  ret void
}

;-----------------------------------------------------------------------------
; All L1 + L2 combinations for stores
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_store_l1_first_l2_first
; CHECK: st.global.L1::evict_first.L2::evict_first.b32
define void @test_store_l1_first_l2_first(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !20
  ret void
}

; CHECK-LABEL: test_store_l1_first_l2_last
; CHECK: st.global.L1::evict_first.L2::evict_last.b32
define void @test_store_l1_first_l2_last(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !21
  ret void
}

; CHECK-LABEL: test_store_l1_last_l2_first
; CHECK: st.global.L1::evict_last.L2::evict_first.b32
define void @test_store_l1_last_l2_first(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !22
  ret void
}

; CHECK-LABEL: test_store_l1_last_l2_last
; CHECK: st.global.L1::evict_last.L2::evict_last.b32
define void @test_store_l1_last_l2_last(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !23
  ret void
}

;-----------------------------------------------------------------------------
; Different data types - loads
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_i16_l1_first
; CHECK: ld.global.L1::evict_first.b16
define i16 @test_load_i16_l1_first(ptr addrspace(1) %p) {
  %v = load i16, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i16 %v
}

; CHECK-LABEL: test_load_i64_l1_last
; CHECK: ld.global.L1::evict_last.b64
define i64 @test_load_i64_l1_last(ptr addrspace(1) %p) {
  %v = load i64, ptr addrspace(1) %p, !mem.cache_hint !1
  ret i64 %v
}

; CHECK-LABEL: test_load_f32_l2_first
; CHECK: ld.global.L2::evict_first.b32
define float @test_load_f32_l2_first(ptr addrspace(1) %p) {
  %v = load float, ptr addrspace(1) %p, !mem.cache_hint !4
  ret float %v
}

; CHECK-LABEL: test_load_f64_l2_last
; CHECK: ld.global.L2::evict_last.b64
define double @test_load_f64_l2_last(ptr addrspace(1) %p) {
  %v = load double, ptr addrspace(1) %p, !mem.cache_hint !5
  ret double %v
}

;-----------------------------------------------------------------------------
; Different data types - stores
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_store_i16_l1_first
; CHECK: st.global.L1::evict_first.b16
define void @test_store_i16_l1_first(ptr addrspace(1) %p, i16 %v) {
  store i16 %v, ptr addrspace(1) %p, !mem.cache_hint !0
  ret void
}

; CHECK-LABEL: test_store_i64_l2_last
; CHECK: st.global.L2::evict_last.b64
define void @test_store_i64_l2_last(ptr addrspace(1) %p, i64 %v) {
  store i64 %v, ptr addrspace(1) %p, !mem.cache_hint !5
  ret void
}

; CHECK-LABEL: test_store_f32_l1_no_allocate
; CHECK: st.global.L1::no_allocate.b32
define void @test_store_f32_l1_no_allocate(ptr addrspace(1) %p, float %v) {
  store float %v, ptr addrspace(1) %p, !mem.cache_hint !3
  ret void
}

;-----------------------------------------------------------------------------
; Vector loads with cache hints
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_v2i32_l1_first
; CHECK: ld.global.L1::evict_first.v2.b32
define <2 x i32> @test_load_v2i32_l1_first(ptr addrspace(1) %p) {
  %v = load <2 x i32>, ptr addrspace(1) %p, !mem.cache_hint !0
  ret <2 x i32> %v
}

; CHECK-LABEL: test_load_v4i32_l2_last
; CHECK: ld.global.L2::evict_last.v4.b32
define <4 x i32> @test_load_v4i32_l2_last(ptr addrspace(1) %p) {
  %v = load <4 x i32>, ptr addrspace(1) %p, !mem.cache_hint !5
  ret <4 x i32> %v
}

; CHECK-LABEL: test_load_v2f32_l1_unchanged
; CHECK: ld.global.L1::evict_unchanged.v2.b32
define <2 x float> @test_load_v2f32_l1_unchanged(ptr addrspace(1) %p) {
  %v = load <2 x float>, ptr addrspace(1) %p, !mem.cache_hint !2
  ret <2 x float> %v
}

; CHECK-LABEL: test_load_v2f64_prefetch_128
; CHECK: ld.global.L2::128B.v2.b64
define <2 x double> @test_load_v2f64_prefetch_128(ptr addrspace(1) %p) {
  %v = load <2 x double>, ptr addrspace(1) %p, !mem.cache_hint !7
  ret <2 x double> %v
}

;-----------------------------------------------------------------------------
; Vector stores with cache hints
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_store_v2i32_l1_last
; CHECK: st.global.L1::evict_last.v2.b32
define void @test_store_v2i32_l1_last(ptr addrspace(1) %p, <2 x i32> %v) {
  store <2 x i32> %v, ptr addrspace(1) %p, !mem.cache_hint !1
  ret void
}

; CHECK-LABEL: test_store_v4i32_l2_first
; CHECK: st.global.L2::evict_first.v4.b32
define void @test_store_v4i32_l2_first(ptr addrspace(1) %p, <4 x i32> %v) {
  store <4 x i32> %v, ptr addrspace(1) %p, !mem.cache_hint !4
  ret void
}

; CHECK-LABEL: test_store_v2f64_l1_no_allocate
; CHECK: st.global.L1::no_allocate.v2.b64
define void @test_store_v2f64_l1_no_allocate(ptr addrspace(1) %p, <2 x double> %v) {
  store <2 x double> %v, ptr addrspace(1) %p, !mem.cache_hint !3
  ret void
}

;-----------------------------------------------------------------------------
; No hint should produce plain load/store
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_no_hint
; CHECK: ld.global.b32
; CHECK-NOT: L1::
; CHECK-NOT: L2::
define i32 @test_load_no_hint(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p
  ret i32 %v
}

; CHECK-LABEL: test_store_no_hint
; CHECK: st.global.b32
; CHECK-NOT: L1::
; CHECK-NOT: L2::
define void @test_store_no_hint(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p
  ret void
}

;-----------------------------------------------------------------------------
; L2::cache_hint with constant cache-policy operand (metadata-based)
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_cache_hint_i32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 12345
; CHECK: ld.global.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i32 @test_load_cache_hint_i32(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !30
  ret i32 %v
}

; CHECK-LABEL: test_load_cache_hint_i64
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 98765
; CHECK: ld.global.L2::cache_hint.b64 {{%rd[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i64 @test_load_cache_hint_i64(ptr addrspace(1) %p) {
  %v = load i64, ptr addrspace(1) %p, !mem.cache_hint !31
  ret i64 %v
}

; CHECK-LABEL: test_load_cache_hint_f32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 55555
; CHECK: ld.global.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define float @test_load_cache_hint_f32(ptr addrspace(1) %p) {
  %v = load float, ptr addrspace(1) %p, !mem.cache_hint !32
  ret float %v
}

; CHECK-LABEL: test_store_cache_hint_i32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 67890
; CHECK: st.global.L2::cache_hint.b32 [{{%rd[0-9]+}}], {{%r[0-9]+}}, [[POLICY]]
define void @test_store_cache_hint_i32(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !33
  ret void
}

; CHECK-LABEL: test_store_cache_hint_i64
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 11111
; CHECK: st.global.L2::cache_hint.b64 [{{%rd[0-9]+}}], {{%rd[0-9]+}}, [[POLICY]]
define void @test_store_cache_hint_i64(ptr addrspace(1) %p, i64 %v) {
  store i64 %v, ptr addrspace(1) %p, !mem.cache_hint !34
  ret void
}

; CHECK-LABEL: test_store_cache_hint_f32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 22222
; CHECK: st.global.L2::cache_hint.b32 [{{%rd[0-9]+}}], {{%r[0-9]+}}, [[POLICY]]
define void @test_store_cache_hint_f32(ptr addrspace(1) %p, float %v) {
  store float %v, ptr addrspace(1) %p, !mem.cache_hint !35
  ret void
}

;-----------------------------------------------------------------------------
; L2::cache_hint with vector types
;-----------------------------------------------------------------------------

; CHECK-LABEL: test_load_cache_hint_v2i32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 33333
; CHECK: ld.global.L2::cache_hint.v2.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}}, [{{%rd[0-9]+}}], [[POLICY]]
define <2 x i32> @test_load_cache_hint_v2i32(ptr addrspace(1) %p) {
  %v = load <2 x i32>, ptr addrspace(1) %p, !mem.cache_hint !40
  ret <2 x i32> %v
}

; CHECK-LABEL: test_load_cache_hint_v4i32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 44444
; CHECK: ld.global.L2::cache_hint.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [{{%rd[0-9]+}}], [[POLICY]]
define <4 x i32> @test_load_cache_hint_v4i32(ptr addrspace(1) %p) {
  %v = load <4 x i32>, ptr addrspace(1) %p, !mem.cache_hint !41
  ret <4 x i32> %v
}

; CHECK-LABEL: test_load_cache_hint_v2i64
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 55556
; CHECK: ld.global.L2::cache_hint.v2.b64 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [{{%rd[0-9]+}}], [[POLICY]]
define <2 x i64> @test_load_cache_hint_v2i64(ptr addrspace(1) %p) {
  %v = load <2 x i64>, ptr addrspace(1) %p, !mem.cache_hint !42
  ret <2 x i64> %v
}

; CHECK-LABEL: test_load_cache_hint_v2f32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 66666
; CHECK: ld.global.L2::cache_hint.v2.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}}, [{{%rd[0-9]+}}], [[POLICY]]
define <2 x float> @test_load_cache_hint_v2f32(ptr addrspace(1) %p) {
  %v = load <2 x float>, ptr addrspace(1) %p, !mem.cache_hint !43
  ret <2 x float> %v
}

; CHECK-LABEL: test_load_cache_hint_v2f64
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 77777
; CHECK: ld.global.L2::cache_hint.v2.b64 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [{{%rd[0-9]+}}], [[POLICY]]
define <2 x double> @test_load_cache_hint_v2f64(ptr addrspace(1) %p) {
  %v = load <2 x double>, ptr addrspace(1) %p, !mem.cache_hint !44
  ret <2 x double> %v
}

; CHECK-LABEL: test_store_cache_hint_v2i32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 88888
; CHECK: st.global.L2::cache_hint.v2.b32 [{{%rd[0-9]+}}], {{{%r[0-9]+}}, {{%r[0-9]+}}}, [[POLICY]]
define void @test_store_cache_hint_v2i32(ptr addrspace(1) %p, <2 x i32> %v) {
  store <2 x i32> %v, ptr addrspace(1) %p, !mem.cache_hint !45
  ret void
}

; CHECK-LABEL: test_store_cache_hint_v4i32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 99999
; CHECK: st.global.L2::cache_hint.v4.b32 [{{%rd[0-9]+}}], {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [[POLICY]]
define void @test_store_cache_hint_v4i32(ptr addrspace(1) %p, <4 x i32> %v) {
  store <4 x i32> %v, ptr addrspace(1) %p, !mem.cache_hint !46
  ret void
}

; CHECK-LABEL: test_store_cache_hint_v2i64
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 11112
; CHECK: st.global.L2::cache_hint.v2.b64 [{{%rd[0-9]+}}], {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [[POLICY]]
define void @test_store_cache_hint_v2i64(ptr addrspace(1) %p, <2 x i64> %v) {
  store <2 x i64> %v, ptr addrspace(1) %p, !mem.cache_hint !47
  ret void
}

; CHECK-LABEL: test_store_cache_hint_v2f32
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 22223
; CHECK: st.global.L2::cache_hint.v2.b32 [{{%rd[0-9]+}}], {{{%r[0-9]+}}, {{%r[0-9]+}}}, [[POLICY]]
define void @test_store_cache_hint_v2f32(ptr addrspace(1) %p, <2 x float> %v) {
  store <2 x float> %v, ptr addrspace(1) %p, !mem.cache_hint !48
  ret void
}

; CHECK-LABEL: test_store_cache_hint_v2f64
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 33334
; CHECK: st.global.L2::cache_hint.v2.b64 [{{%rd[0-9]+}}], {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [[POLICY]]
define void @test_store_cache_hint_v2f64(ptr addrspace(1) %p, <2 x double> %v) {
  store <2 x double> %v, ptr addrspace(1) %p, !mem.cache_hint !49
  ret void
}

;-----------------------------------------------------------------------------
; L2::cache_hint combined with other hints (L2::cache_hint takes precedence)
;-----------------------------------------------------------------------------

; L2::cache_hint + L1 eviction: both qualifiers should be emitted
; CHECK-LABEL: test_load_cache_hint_with_l1
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 44445
; CHECK: ld.global.L1::evict_first.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i32 @test_load_cache_hint_with_l1(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !50
  ret i32 %v
}

; L2::cache_hint + L2 eviction: both qualifiers should be emitted
; CHECK-LABEL: test_load_cache_hint_with_l2_eviction
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 55557
; CHECK: ld.global.L2::evict_last.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i32 @test_load_cache_hint_with_l2_eviction(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !51
  ret i32 %v
}

; L2::cache_hint + L2 prefetch: both qualifiers should be emitted
; CHECK-LABEL: test_load_cache_hint_with_prefetch
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 66667
; CHECK: ld.global.L2::cache_hint.L2::128B.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i32 @test_load_cache_hint_with_prefetch(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !52
  ret i32 %v
}

; L2::cache_hint + all other hints: all qualifiers emitted
; CHECK-LABEL: test_load_cache_hint_with_all
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 77778
; CHECK: ld.global.L1::evict_last.L2::evict_first.L2::cache_hint.L2::256B.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i32 @test_load_cache_hint_with_all(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !53
  ret i32 %v
}

; Store: L2::cache_hint + L1 eviction
; CHECK-LABEL: test_store_cache_hint_with_l1
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 88889
; CHECK: st.global.L1::evict_unchanged.L2::cache_hint.b32 [{{%rd[0-9]+}}], {{%r[0-9]+}}, [[POLICY]]
define void @test_store_cache_hint_with_l1(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !54
  ret void
}

; Store: L2::cache_hint + L1 + L2 eviction (all qualifiers emitted)
; CHECK-LABEL: test_store_cache_hint_with_all
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 99990
; CHECK: st.global.L1::no_allocate.L2::evict_last.L2::cache_hint.b32 [{{%rd[0-9]+}}], {{%r[0-9]+}}, [[POLICY]]
define void @test_store_cache_hint_with_all(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !55
  ret void
}

; Vector load: L2::cache_hint + L1 eviction
; CHECK-LABEL: test_load_cache_hint_v2i32_with_l1
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 11113
; CHECK: ld.global.L1::evict_first.L2::cache_hint.v2.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}}, [{{%rd[0-9]+}}], [[POLICY]]
define <2 x i32> @test_load_cache_hint_v2i32_with_l1(ptr addrspace(1) %p) {
  %v = load <2 x i32>, ptr addrspace(1) %p, !mem.cache_hint !56
  ret <2 x i32> %v
}

; Vector store: L2::cache_hint + L1 + L2 eviction + prefetch (all qualifiers emitted)
; CHECK-LABEL: test_store_cache_hint_v2i32_with_all
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 22224
; CHECK: st.global.L1::evict_last.L2::evict_first.L2::cache_hint.L2::64B.v2.b32 [{{%rd[0-9]+}}], {{{%r[0-9]+}}, {{%r[0-9]+}}}, [[POLICY]]
define void @test_store_cache_hint_v2i32_with_all(ptr addrspace(1) %p, <2 x i32> %v) {
  store <2 x i32> %v, ptr addrspace(1) %p, !mem.cache_hint !57
  ret void
}

;-----------------------------------------------------------------------------
; Multiple loads sharing same pointer - each gets its own policy
;-----------------------------------------------------------------------------

; Two volatile loads from the same pointer - each load gets its own cache policy.
; The per-MMO storage ensures no policy collisions when multiple memops share
; the same pointer operand but have different cache policies.
; CHECK-LABEL: test_multiple_loads_same_ptr
; CHECK-DAG: mov.b64 [[POLICY1:%rd[0-9]+]], 11111
; CHECK-DAG: mov.b64 [[POLICY2:%rd[0-9]+]], 22222
; CHECK-DAG: ld.volatile.global.L1::evict_last.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY1]]
; CHECK-DAG: ld.volatile.global.L1::evict_first.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY2]]
define i32 @test_multiple_loads_same_ptr(ptr addrspace(1) %p) {
  %v1 = load volatile i32, ptr addrspace(1) %p, !mem.cache_hint !60
  %v2 = load volatile i32, ptr addrspace(1) %p, !mem.cache_hint !61
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

;-----------------------------------------------------------------------------
; Invalid/edge cases
;-----------------------------------------------------------------------------

; Test with invalid operand_no - should be ignored
; CHECK-LABEL: test_load_invalid_operand_no
; CHECK: ld.global.b32
; CHECK-NOT: L1::
define i32 @test_load_invalid_operand_no(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !11
  ret i32 %v
}

; Test with unknown key - should be ignored, but valid L1 hint should still work
; CHECK-LABEL: test_load_unknown_key
; CHECK: ld.global.L1::evict_first.b32
define i32 @test_load_unknown_key(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !12
  ret i32 %v
}

; Test with reordered metadata (operand_no not first) - should still work
; CHECK-LABEL: test_load_reordered_metadata
; CHECK: ld.global.L1::evict_last.L2::evict_first.b32
define i32 @test_load_reordered_metadata(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !70
  ret i32 %v
}

;-----------------------------------------------------------------------------
; nvvm.l2_cache_hint with invalid value - should NOT emit L2::cache_hint
; These tests verify that when nvvm.l2_cache_hint key exists but the value
; is not a valid i64 constant, we do NOT emit L2::cache_hint mode.
;-----------------------------------------------------------------------------

; nvvm.l2_cache_hint with string value instead of i64 - should be ignored
; CHECK-LABEL: test_load_cache_hint_string_value
; CHECK: ld.global.L1::evict_first.b32 %r{{[0-9]+}}, [%rd{{[0-9]+}}];
; CHECK-NOT: L2::cache_hint
define i32 @test_load_cache_hint_string_value(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !80
  ret i32 %v
}

; nvvm.l2_cache_hint with null/missing value - should be ignored
; CHECK-LABEL: test_load_cache_hint_null_value
; CHECK: ld.global.L1::evict_last.b32 %r{{[0-9]+}}, [%rd{{[0-9]+}}];
; CHECK-NOT: L2::cache_hint
define i32 @test_load_cache_hint_null_value(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !81
  ret i32 %v
}

; nvvm.l2_cache_hint with wrong type (i32 instead of i64) - should still work
; as mdconst::dyn_extract<ConstantInt> accepts any integer type
; CHECK-LABEL: test_load_cache_hint_i32_value
; CHECK: mov.b64 [[POLICY:%rd[0-9]+]], 99999
; CHECK: ld.global.L2::cache_hint.b32 {{%r[0-9]+}}, [{{%rd[0-9]+}}], [[POLICY]]
define i32 @test_load_cache_hint_i32_value(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !82
  ret i32 %v
}

; Store: nvvm.l2_cache_hint with string value - should be ignored
; CHECK-LABEL: test_store_cache_hint_string_value
; CHECK: st.global.L1::evict_unchanged.b32 [%rd{{[0-9]+}}], %r{{[0-9]+}};
; CHECK-NOT: L2::cache_hint
define void @test_store_cache_hint_string_value(ptr addrspace(1) %p, i32 %v) {
  store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !83
  ret void
}

; Test "normal" eviction - should not emit any qualifier (default behavior)
; CHECK-LABEL: test_load_l1_normal
; CHECK: ld.global.b32
; CHECK-NOT: L1::evict_normal
define i32 @test_load_l1_normal(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !13
  ret i32 %v
}

; CHECK-LABEL: test_load_l2_normal
; CHECK: ld.global.b32
; CHECK-NOT: L2::evict_normal
define i32 @test_load_l2_normal(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !14
  ret i32 %v
}

;-----------------------------------------------------------------------------
; Metadata definitions
;-----------------------------------------------------------------------------

; L1 eviction policies
!0 = !{!100}
!100 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"first"}

!1 = !{!101}
!101 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"last"}

!2 = !{!102}
!102 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"unchanged"}

!3 = !{!103}
!103 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"no_allocate"}

; L2 eviction policies
!4 = !{!104}
!104 = !{!"operand_no", i32 0, !"nvvm.l2_eviction", !"first"}

!5 = !{!105}
!105 = !{!"operand_no", i32 0, !"nvvm.l2_eviction", !"last"}

; L2 prefetch sizes
!6 = !{!106}
!106 = !{!"operand_no", i32 0, !"nvvm.l2_prefetch_size", !"64B"}

!7 = !{!107}
!107 = !{!"operand_no", i32 0, !"nvvm.l2_prefetch_size", !"128B"}

!8 = !{!108}
!108 = !{!"operand_no", i32 0, !"nvvm.l2_prefetch_size", !"256B"}

; Invalid operand_no (should be ignored for load which has operand 0)
!11 = !{!111}
!111 = !{!"operand_no", i32 5, !"nvvm.l1_eviction", !"first"}

; Unknown key (should be ignored, but valid L1 hint should still work)
!12 = !{!112}
!112 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"first", !"nvvm.unknown_key", !"value"}

; "normal" eviction (default, should not emit qualifier)
!13 = !{!113}
!113 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"normal"}

!14 = !{!114}
!114 = !{!"operand_no", i32 0, !"nvvm.l2_eviction", !"normal"}

; All L1 + L2 combinations
!20 = !{!120}
!120 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"first"}

!21 = !{!121}
!121 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"last"}

!22 = !{!122}
!122 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"first"}

!23 = !{!123}
!123 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"last"}

; L1 + L2 + Prefetch combination
!24 = !{!124}
!124 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"first", !"nvvm.l2_eviction", !"last", !"nvvm.l2_prefetch_size", !"128B"}

; L2::cache_hint with constant cache-policy
!30 = !{!130}
!130 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 12345}

!31 = !{!131}
!131 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 98765}

!32 = !{!132}
!132 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 55555}

!33 = !{!133}
!133 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 67890}

!34 = !{!134}
!134 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 11111}

!35 = !{!135}
!135 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 22222}

; L2::cache_hint for vector types
!40 = !{!140}
!140 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 33333}

!41 = !{!141}
!141 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 44444}

!42 = !{!142}
!142 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 55556}

!43 = !{!143}
!143 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 66666}

!44 = !{!144}
!144 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 77777}

!45 = !{!145}
!145 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 88888}

!46 = !{!146}
!146 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 99999}

!47 = !{!147}
!147 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 11112}

!48 = !{!148}
!148 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 22223}

!49 = !{!149}
!149 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 33334}

; L2::cache_hint combined with other hints (L2::cache_hint takes precedence)
!50 = !{!150}
!150 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 44445, !"nvvm.l1_eviction", !"first"}

!51 = !{!151}
!151 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 55557, !"nvvm.l2_eviction", !"last"}

!52 = !{!152}
!152 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 66667, !"nvvm.l2_prefetch_size", !"128B"}

!53 = !{!153}
!153 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 77778, !"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"first", !"nvvm.l2_prefetch_size", !"256B"}

!54 = !{!154}
!154 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 88889, !"nvvm.l1_eviction", !"unchanged"}

!55 = !{!155}
!155 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 99990, !"nvvm.l1_eviction", !"no_allocate", !"nvvm.l2_eviction", !"last"}

!56 = !{!156}
!156 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 11113, !"nvvm.l1_eviction", !"first"}

!57 = !{!157}
!157 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 22224, !"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"first", !"nvvm.l2_prefetch_size", !"64B"}

; Multiple loads same pointer test (different policies)
!60 = !{!160}
!160 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 11111, !"nvvm.l1_eviction", !"last"}

!61 = !{!161}
!161 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i64 22222, !"nvvm.l1_eviction", !"first"}

; Reordered metadata test (operand_no not first)
!70 = !{!170}
!170 = !{!"nvvm.l1_eviction", !"last", !"nvvm.l2_eviction", !"first", !"operand_no", i32 0}

; Invalid nvvm.l2_cache_hint values - should be ignored, no L2::cache_hint emitted
; String value instead of i64 - invalid, L1 hint should still work
!80 = !{!180}
!180 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"first", !"nvvm.l2_cache_hint", !"not_a_number"}

; Null/metadata reference instead of constant - invalid
!81 = !{!181}
!181 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"last", !"nvvm.l2_cache_hint", !{}}

; i32 instead of i64 - still valid, ConstantInt accepts any integer type
!82 = !{!182}
!182 = !{!"operand_no", i32 0, !"nvvm.l2_cache_hint", i32 99999}

; Store: string value for nvvm.l2_cache_hint - invalid
!83 = !{!183}
!183 = !{!"operand_no", i32 0, !"nvvm.l1_eviction", !"unchanged", !"nvvm.l2_cache_hint", !"invalid"}
