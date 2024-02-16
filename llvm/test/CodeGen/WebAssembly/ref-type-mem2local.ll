; RUN: opt < %s -wasm-ref-type-mem2local -S | FileCheck %s

target triple = "wasm32-unknown-unknown"

%externref = type ptr addrspace(10)
%funcref = type ptr addrspace(20)

declare %funcref @get_funcref()
declare %externref @get_externref()
declare void @take_funcref(%funcref)
declare void @take_externref(%externref)

; CHECK-LABEL: @test_ref_type_mem2local
define void @test_ref_type_mem2local() {
entry:
  %alloc.externref = alloca %externref, align 1
  %eref = call %externref @get_externref()
  store %externref %eref, ptr %alloc.externref, align 1
  %eref.loaded = load %externref, ptr %alloc.externref, align 1
  call void @take_externref(%externref %eref.loaded)
  ; CHECK:      %alloc.externref.var = alloca ptr addrspace(10), align 1, addrspace(1)
  ; CHECK-NEXT: %eref = call ptr addrspace(10) @get_externref()
  ; CHECK-NEXT: store ptr addrspace(10) %eref, ptr addrspace(1) %alloc.externref.var, align 1
  ; CHECK-NEXT: %eref.loaded = load ptr addrspace(10), ptr addrspace(1) %alloc.externref.var, align 1
  ; CHECK-NEXT: call void @take_externref(ptr addrspace(10) %eref.loaded)

  %alloc.funcref = alloca %funcref, align 1
  %fref = call %funcref @get_funcref()
  store %funcref %fref, ptr %alloc.funcref, align 1
  %fref.loaded = load %funcref, ptr %alloc.funcref, align 1
  call void @take_funcref(%funcref %fref.loaded)
  ; CHECK-NEXT: %alloc.funcref.var = alloca ptr addrspace(20), align 1, addrspace(1)
  ; CHECK-NEXT: %fref = call ptr addrspace(20) @get_funcref()
  ; CHECK-NEXT: store ptr addrspace(20) %fref, ptr addrspace(1) %alloc.funcref.var, align 1
  ; CHECK-NEXT: %fref.loaded = load ptr addrspace(20), ptr addrspace(1) %alloc.funcref.var, align 1
  ; CHECK-NEXT: call void @take_funcref(ptr addrspace(20) %fref.loaded)

  ret void
}
