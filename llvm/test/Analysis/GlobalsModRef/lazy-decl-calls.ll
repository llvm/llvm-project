; RUN: opt < %s -aa-pipeline=globals-aa -passes='require<globals-aa>,dse' -S | FileCheck %s
;
; Direct calls to declarations use attribute modeling; readnone must not
; block DSE, unknown decls and indirect calls stay conservative. Double call
; covers DeclCallees dedup.

@X = internal global i32 0

define void @test_readnone() {
; CHECK-LABEL: @test_readnone
; CHECK-NOT: store i32 1, ptr @X
; CHECK: call void @decl_readnone()
; CHECK-NEXT: store i32 2, ptr @X
  store i32 1, ptr @X
  call void @decl_readnone()
  store i32 2, ptr @X
  ret void
}

define void @test_readnone_twice() {
; CHECK-LABEL: @test_readnone_twice
; CHECK: call void @decl_readnone()
; CHECK-NEXT: call void @decl_readnone()
; CHECK-NEXT: store i32 4, ptr @X
  store i32 3, ptr @X
  call void @decl_readnone()
  call void @decl_readnone()
  store i32 4, ptr @X
  ret void
}

define void @test_unknown() {
; CHECK-LABEL: @test_unknown
; CHECK: store i32 5, ptr @X
; CHECK-NEXT: call void @decl_unknown()
; CHECK-NEXT: store i32 6, ptr @X
  store i32 5, ptr @X
  call void @decl_unknown()
  store i32 6, ptr @X
  ret void
}

define void @test_indirect(ptr %fp) {
; CHECK-LABEL: @test_indirect
; CHECK: store i32 7, ptr @X
; CHECK-NEXT: call void %fp()
; CHECK-NEXT: store i32 8, ptr @X
  store i32 7, ptr @X
  call void %fp()
  store i32 8, ptr @X
  ret void
}

declare void @decl_readnone() readnone nounwind
declare void @decl_unknown()
