; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-COUNT-9: %[[#]] = OpAtomicExchange %[[#Long]] %[[#]] %[[#]] %[[#]] %[[#]]

%Type1 = type { i64 }
%Type2 = type { ptr addrspace(4) }

define linkonce_odr dso_local spir_func void @f1() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePyN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEy(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}

define linkonce_odr dso_local spir_func void @f2() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePxN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEx(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}

define linkonce_odr dso_local spir_func void @f3() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}

define linkonce_odr dso_local spir_func void @f4() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePlN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEl(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}


define linkonce_odr dso_local spir_func void @f5() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

define linkonce_odr dso_local spir_func void @f6() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

define linkonce_odr dso_local spir_func void @f7() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

define linkonce_odr dso_local spir_func void @f8() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func void @f9() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePyN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEy(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePxN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEx(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePlN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEl(ptr addrspace(4), i32, i32, i64)
