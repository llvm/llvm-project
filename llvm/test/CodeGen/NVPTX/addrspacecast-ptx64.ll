; RUN: llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_80 | FileCheck %s -check-prefixes=ALL,NOPTRCONV,CLS64
; RUN: llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_80 --nvptx-short-ptr | FileCheck %s -check-prefixes=ALL,PTRCONV,CLS64
; RUN: %if ptxas-12.8 %{ llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_80 | %ptxas-verify %}
; RUN: %if ptxas-12.8 %{ llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_80 --nvptx-short-ptr | %ptxas-verify %}

; ALL-LABEL: conv_shared_cluster_to_generic
define i32 @conv_shared_cluster_to_generic(ptr addrspace(7) %ptr) {
; CLS32: cvta.shared::cluster.u32
; PTRCONV: cvt.u64.u32
; NOPTRCONV-NOT: cvt.u64.u32
; CLS64: cvta.shared::cluster.u64
; ALL: ld.u32
  %genptr = addrspacecast ptr addrspace(7) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv_generic_to_shared_cluster
define i32 @conv_generic_to_shared_cluster(ptr %ptr) {
; CLS32: cvta.to.shared::cluster.u32
; CLS64: cvta.to.shared::cluster.u64
; PTRCONV: cvt.u32.u64
; NOPTRCONV-NOT: cvt.u32.u64
; ALL: ld.shared::cluster.u32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(7)
  %val = load i32, ptr addrspace(7) %specptr
  ret i32 %val
}

; ALL-LABEL: conv_shared_to_shared_cluster
define i32 @conv_shared_to_shared_cluster(ptr addrspace(3) %ptr) {
; CLS64: cvta.shared.u64 
; CLS64: cvta.to.shared::cluster.u64
; ALL: ld.shared::cluster.u32
  %specptr = addrspacecast ptr addrspace(3) %ptr to ptr addrspace(7)
  %val = load i32, ptr addrspace(7) %specptr
  ret i32 %val
}

; ALL-LABEL: conv_shared_cluster_to_shared
define i32 @conv_shared_cluster_to_shared(ptr addrspace(7) %ptr) {
; CLS64: cvta.shared::cluster.u64
; CLS64: cvta.to.shared.u64 
; ALL: ld.shared.u32
  %specptr = addrspacecast ptr addrspace(7) %ptr to ptr addrspace(3)
  %val = load i32, ptr addrspace(3) %specptr
  ret i32 %val
}