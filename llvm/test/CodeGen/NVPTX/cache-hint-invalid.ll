; RUN: split-file %s %t
; RUN: llc < %t/bad-empty.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-EMPTY
; RUN: llc < %t/bad-key.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-KEY
; RUN: llc < %t/bad-other-key.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-OTHER-KEY
; RUN: llc < %t/bad-l1-type.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-L1-TYPE
; RUN: llc < %t/bad-l1-value.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-L1-VALUE
; RUN: llc < %t/bad-l2-type.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-L2-TYPE
; RUN: llc < %t/bad-l2-value.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-L2-VALUE
; RUN: llc < %t/bad-prefetch-type.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-PREFETCH-TYPE
; RUN: llc < %t/bad-prefetch-value.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-PREFETCH-VALUE
; RUN: llc < %t/bad-policy.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx74 2>&1 | FileCheck %s --check-prefix=BAD-POLICY

;--- bad-empty.ll

; Test with empty hint node - should produce an empty-node warning.
; BAD-EMPTY: warning: invalid NVPTX !mem.cache_hint metadata: empty hint node
define i32 @bad_empty_hint_node(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{}

;--- bad-key.ll

; Test with misspelled NVPTX key - should produce an unknown-key warning.
; BAD-KEY: warning: invalid NVPTX !mem.cache_hint metadata: unknown key 'nvvm.l1_evict'
define i32 @bad_nvvm_key(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l1_evict", !"first"}

;--- bad-other-key.ll

; Test with a non-NVPTX key - should produce an unknown-key warning.
; BAD-OTHER-KEY: warning: invalid NVPTX !mem.cache_hint metadata: unknown key 'some.target_hint'
define i32 @bad_other_key(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"some.target_hint", !"value"}

;--- bad-l1-type.ll

; nvvm.l1_eviction expects a string value.
; BAD-L1-TYPE: warning: invalid NVPTX !mem.cache_hint metadata: 'nvvm.l1_eviction' expects a string value
define i32 @bad_l1_type(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l1_eviction", i32 0}

;--- bad-l1-value.ll

; nvvm.l1_eviction accepts only known PTX eviction values.
; BAD-L1-VALUE: warning: invalid NVPTX !mem.cache_hint metadata: unknown value 'middle' for 'nvvm.l1_eviction'
define i32 @bad_l1_value(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l1_eviction", !"middle"}

;--- bad-l2-type.ll

; nvvm.l2_eviction expects a string value.
; BAD-L2-TYPE: warning: invalid NVPTX !mem.cache_hint metadata: 'nvvm.l2_eviction' expects a string value
define i32 @bad_l2_type(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l2_eviction", i32 0}

;--- bad-l2-value.ll

; nvvm.l2_eviction accepts only known PTX eviction values.
; BAD-L2-VALUE: warning: invalid NVPTX !mem.cache_hint metadata: unknown value 'middle' for 'nvvm.l2_eviction'
define i32 @bad_l2_value(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l2_eviction", !"middle"}

;--- bad-prefetch-type.ll

; nvvm.l2_prefetch_size expects a string value.
; BAD-PREFETCH-TYPE: warning: invalid NVPTX !mem.cache_hint metadata: 'nvvm.l2_prefetch_size' expects a string value
define i32 @bad_prefetch_type(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l2_prefetch_size", i32 64}

;--- bad-prefetch-value.ll

; nvvm.l2_prefetch_size accepts only supported PTX prefetch sizes.
; BAD-PREFETCH-VALUE: warning: invalid NVPTX !mem.cache_hint metadata: unknown value '32B' for 'nvvm.l2_prefetch_size'
define i32 @bad_prefetch_value(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l2_prefetch_size", !"32B"}

;--- bad-policy.ll

; nvvm.l2_cache_hint expects an integer cache-policy value.
; BAD-POLICY: warning: invalid NVPTX !mem.cache_hint metadata: 'nvvm.l2_cache_hint' expects an integer value
define i32 @bad_cache_policy_value(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p, !mem.cache_hint !0
  ret i32 %v
}

!0 = !{i32 0, !1}
!1 = !{!"nvvm.l2_cache_hint", !"not_an_integer"}
