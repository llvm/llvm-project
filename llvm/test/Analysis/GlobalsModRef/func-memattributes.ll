; RUN: opt < %s -aa-pipeline=globals-aa -passes='require<globals-aa>,dse' -S | FileCheck %s

@X = internal global i32 4

define i32 @test0() {
; CHECK-LABEL: @test0
; CHECK: store i32 0, i32* @X
; CHECK-NEXT: call i32 @func_readonly() #0
; CHECK-NEXT: store i32 1, i32* @X
  store i32 0, i32* @X
  %x = call i32 @func_readonly() #0
  store i32 1, i32* @X
  ret i32 %x
}

define i32 @test1() {
; CHECK-LABEL: @test1
; CHECK-NOT: store
; CHECK: call i32 @func_read_argmem_only() #1
; CHECK-NEXT: store i32 3, i32* @X
  store i32 2, i32* @X
  %x = call i32 @func_read_argmem_only() #1
  store i32 3, i32* @X
  ret i32 %x
}

declare i32 @func_readonly() #0
declare i32 @func_read_argmem_only() #1

attributes #0 = { readonly nounwind }
attributes #1 = { readonly argmemonly nounwind }
