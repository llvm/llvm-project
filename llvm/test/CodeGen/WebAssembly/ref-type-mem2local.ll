; RUN: llc < %s -mattr=+reference-types -stop-after=wasm-ref-type-mem2local | FileCheck %s
; RUN: llc < %s -stop-after=wasm-ref-type-mem2local | FileCheck %s --check-prefix=ATTR

target triple = "wasm32-unknown-unknown"

%externref = type target("wasm.externref")
%funcref = type target("wasm.funcref")

declare %externref @get_externref()
declare %funcref @get_funcref()
declare i32 @get_i32()
declare void @take_externref(%externref)
declare void @take_funcref(%funcref)
declare void @take_i32(i32)

; Reference type allocas should be moved to addrspace(1)
; CHECK-LABEL: @test_ref_type_mem2local
define void @test_ref_type_mem2local() {
entry:
  %alloc.externref = alloca %externref, align 1
  %eref = call %externref @get_externref()
  store %externref %eref, ptr %alloc.externref, align 1
  %eref.loaded = load %externref, ptr %alloc.externref, align 1
  call void @take_externref(%externref %eref.loaded)
  ; CHECK:      %alloc.externref.var = alloca target("wasm.externref"), align 1, addrspace(1)
  ; CHECK-NEXT: %eref = call target("wasm.externref") @get_externref()
  ; CHECK-NEXT: store target("wasm.externref") %eref, ptr addrspace(1) %alloc.externref.var, align 1
  ; CHECK-NEXT: %eref.loaded = load target("wasm.externref"), ptr addrspace(1) %alloc.externref.var, align 1
  ; CHECK-NEXT: call void @take_externref(target("wasm.externref") %eref.loaded)

  %alloc.funcref = alloca %funcref, align 1
  %fref = call %funcref @get_funcref()
  store %funcref %fref, ptr %alloc.funcref, align 1
  %fref.loaded = load %funcref, ptr %alloc.funcref, align 1
  call void @take_funcref(%funcref %fref.loaded)
  ; CHECK-NEXT: %alloc.funcref.var = alloca target("wasm.funcref"), align 1, addrspace(1)
  ; CHECK-NEXT: %fref = call target("wasm.funcref") @get_funcref()
  ; CHECK-NEXT: store target("wasm.funcref") %fref, ptr addrspace(1) %alloc.funcref.var, align 1
  ; CHECK-NEXT: %fref.loaded = load target("wasm.funcref"), ptr addrspace(1) %alloc.funcref.var, align 1
  ; CHECK-NEXT: call void @take_funcref(target("wasm.funcref") %fref.loaded)

  ret void
}

; POD type allocas should stay the same
; CHECK-LABEL: @test_pod_type
define void @test_pod_type() {
entry:
  %alloc.i32 = alloca i32
  %i32 = call i32 @get_i32()
  store i32 %i32, ptr %alloc.i32
  %i32.loaded = load i32, ptr %alloc.i32
  call void @take_i32(i32 %i32.loaded)
  ; CHECK: %alloc.i32 = alloca i32, align 4{{$}}
  ; CHECK-NOT: alloca i32 {{.*}} addrspace(1)

  ret void
}

; The same function as test_ref_type_mem2local, but here +reference-types is
; given in the function attribute.
; Reference type allocas should be moved to addrspace(1)
; ATTR-LABEL: @test_ref_type_mem2local_func_attr
define void @test_ref_type_mem2local_func_attr() #0 {
entry:
  %alloc.externref = alloca %externref, align 1
  %eref = call %externref @get_externref()
  store %externref %eref, ptr %alloc.externref, align 1
  %eref.loaded = load %externref, ptr %alloc.externref, align 1
  call void @take_externref(%externref %eref.loaded)
  ; ATTR:      %alloc.externref.var = alloca target("wasm.externref"), align 1, addrspace(1)
  ; ATTR-NEXT: %eref = call target("wasm.externref") @get_externref()
  ; ATTR-NEXT: store target("wasm.externref") %eref, ptr addrspace(1) %alloc.externref.var, align 1
  ; ATTR-NEXT: %eref.loaded = load target("wasm.externref"), ptr addrspace(1) %alloc.externref.var, align 1
  ; ATTR-NEXT: call void @take_externref(target("wasm.externref") %eref.loaded)

  %alloc.funcref = alloca %funcref, align 1
  %fref = call %funcref @get_funcref()
  store %funcref %fref, ptr %alloc.funcref, align 1
  %fref.loaded = load %funcref, ptr %alloc.funcref, align 1
  call void @take_funcref(%funcref %fref.loaded)
  ; ATTR-NEXT: %alloc.funcref.var = alloca target("wasm.funcref"), align 1, addrspace(1)
  ; ATTR-NEXT: %fref = call target("wasm.funcref") @get_funcref()
  ; ATTR-NEXT: store target("wasm.funcref") %fref, ptr addrspace(1) %alloc.funcref.var, align 1
  ; ATTR-NEXT: %fref.loaded = load target("wasm.funcref"), ptr addrspace(1) %alloc.funcref.var, align 1
  ; ATTR-NEXT: call void @take_funcref(target("wasm.funcref") %fref.loaded)

  ret void
}

attributes #0 = { "target-features"="+reference-types" }
