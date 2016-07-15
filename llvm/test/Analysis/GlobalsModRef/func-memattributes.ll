; RUN: opt < %s -disable-basicaa -globals-aa -dse -S | FileCheck %s

@X = internal global i32 4

define void @test0() {
; CHECK-LABEL: @test0
; CHECK: store i32 0, i32* @X
; CHECK-NEXT: call void @func_readonly() #0
; CHECK-NEXT: store i32 1, i32* @X
  store i32 0, i32* @X
  call void @func_readonly() #0
  store i32 1, i32* @X
  ret void
}

define void @test1() {
; CHECK-LABEL: @test1
; CHECK-NOT: store
; CHECK: call void @func_read_argmem_only() #1
; CHECK-NEXT: store i32 3, i32* @X
  store i32 2, i32* @X
  call void @func_read_argmem_only() #1
  store i32 3, i32* @X
  ret void
}

declare void @func_readonly() #0
declare void @func_read_argmem_only() #1

attributes #0 = { readonly }
attributes #1 = { readonly argmemonly }
