; NOTE: Do not autogenerate. These checks test inline-cost diagnostics, not output IR.
; REQUIRES: asserts
; RUN: opt -mtriple=amdgpu9.50-amd-amdhsa -passes=inline -inline-threshold=100 -inline-call-penalty=0 -inline-cost-full=true -debug-only=inline-cost -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -mtriple=amdgpu9.50-amd-amdhsa -passes=inline -inline-threshold=100 -inline-call-penalty=0 -inline-cost-full=true -debug-only=inline-cost -amdgpu-inline-arg-lds-cost=7 -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CAPPED
; RUN: opt -mtriple=amdgpu9.50-amd-amdhsa -passes=inline -inline-threshold=100 -inline-call-penalty=0 -inline-cost-full=true -debug-only=inline-cost -amdgpu-inline-arg-lds-cost=0 -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,OFF

; Each ordinary access contributes 5 before the AMDGPU multiplier of 11 and
; single-block bonus of 50%. Cap 7 also tests a partial final access credit.

define void @store_pointer(ptr %dest, ptr %value) {
  store ptr %value, ptr %dest, align 8
  ret void
}

define void @store_self(ptr %dest) {
  store ptr %dest, ptr %dest, align 8
  ret void
}

; Count one access even when its address is also its stored value.
; CHECK-LABEL: Analyzing call of store_pointer... (caller:ordinary_store)
; DEFAULT: Threshold: 1732
; CAPPED: Threshold: 1732
; OFF: Threshold: 1650
define void @ordinary_store(ptr addrspace(3) %dest, ptr %value) {
  %flat = addrspacecast ptr addrspace(3) %dest to ptr
  call void @store_pointer(ptr %flat, ptr %value)
  ret void
}

; CHECK-LABEL: Analyzing call of store_self... (caller:self_store)
; DEFAULT: Threshold: 1732
; CAPPED: Threshold: 1732
; OFF: Threshold: 1650
define void @self_store(ptr addrspace(3) %dest) {
  %flat = addrspacecast ptr addrspace(3) %dest to ptr
  call void @store_self(ptr %flat)
  ret void
}

; Storing an LDS pointer through an unknown address earns no credit.
; CHECK-LABEL: Analyzing call of store_pointer... (caller:value_only)
; DEFAULT: Threshold: 1650
; CAPPED: Threshold: 1650
; OFF: Threshold: 1650
define void @value_only(ptr %dest, ptr addrspace(3) %value) {
  %flat = addrspacecast ptr addrspace(3) %value to ptr
  call void @store_pointer(ptr %dest, ptr %flat)
  ret void
}

define void @copy_sum(ptr %src, ptr %dest) {
  %first = load i32, ptr %src, align 4
  %next = getelementptr i32, ptr %src, i32 1
  %second = load i32, ptr %next, align 4
  %sum = add i32 %first, %second
  store i32 %sum, ptr %dest, align 4
  ret void
}

; Both loads count, including the one reached through a GEP.
; CHECK-LABEL: Analyzing call of copy_sum... (caller:shared_source)
; DEFAULT: Threshold: 1815
; CAPPED: Threshold: 1765
; OFF: Threshold: 1650
define void @shared_source(ptr addrspace(3) %src, ptr %dest) {
  %flat = addrspacecast ptr addrspace(3) %src to ptr
  call void @copy_sum(ptr %flat, ptr %dest)
  ret void
}

; The destination contributes one store independently of the source.
; CHECK-LABEL: Analyzing call of copy_sum... (caller:shared_destination)
; DEFAULT: Threshold: 1732
; CAPPED: Threshold: 1732
; OFF: Threshold: 1650
define void @shared_destination(ptr %src, ptr addrspace(3) %dest) {
  %flat = addrspacecast ptr addrspace(3) %dest to ptr
  call void @copy_sum(ptr %src, ptr %flat)
  ret void
}

; Three accesses across two arguments share one per-call cap.
; CHECK-LABEL: Analyzing call of copy_sum... (caller:shared_both)
; DEFAULT: Threshold: 1897
; CAPPED: Threshold: 1765
; OFF: Threshold: 1650
define void @shared_both(ptr addrspace(3) %src, ptr addrspace(3) %dest) {
  %flat.src = addrspacecast ptr addrspace(3) %src to ptr
  %flat.dest = addrspacecast ptr addrspace(3) %dest to ptr
  call void @copy_sum(ptr %flat.src, ptr %flat.dest)
  ret void
}
