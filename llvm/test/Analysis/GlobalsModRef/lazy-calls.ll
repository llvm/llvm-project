; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='cgscc(inline),require<globals-aa>,dse' -S | FileCheck %s
;
; Effects of declarations and indirect calls must come from the current IR,
; including calls gained by inlining. Callees are nounwind because may-throw
; calls keep stores regardless of AA.

@X = internal global i32 0

declare void @decl_unknown()
declare void @decl_readnone() readnone nounwind

define void @calls_unknown() noinline nounwind {
  call void @decl_unknown()
  ret void
}

define void @test_unknown() {
; CHECK-LABEL: define void @test_unknown(
; CHECK: store i32 1, ptr @X
; CHECK-NEXT: call void @calls_unknown()
; CHECK-NEXT: store i32 2, ptr @X
  store i32 1, ptr @X
  call void @calls_unknown()
  store i32 2, ptr @X
  ret void
}

define void @calls_indirect(ptr %fp) noinline nounwind {
  call void %fp()
  ret void
}

define void @test_indirect(ptr %fp) {
; CHECK-LABEL: define void @test_indirect(
; CHECK: store i32 3, ptr @X
; CHECK-NEXT: call void @calls_indirect(ptr %fp)
; CHECK-NEXT: store i32 4, ptr @X
  store i32 3, ptr @X
  call void @calls_indirect(ptr %fp)
  store i32 4, ptr @X
  ret void
}

define void @calls_readnone_twice() noinline nounwind {
  call void @decl_readnone()
  call void @decl_readnone()
  ret void
}

define void @test_readnone() {
; CHECK-LABEL: define void @test_readnone(
; CHECK-NOT: store i32 5, ptr @X
; CHECK: call void @calls_readnone_twice()
; CHECK-NEXT: store i32 6, ptr @X
  store i32 5, ptr @X
  call void @calls_readnone_twice()
  store i32 6, ptr @X
  ret void
}

define internal void @unknown_callee() {
  call void @decl_unknown()
  ret void
}

define void @inlines_unknown() noinline nounwind {
; CHECK-LABEL: define void @inlines_unknown(
; CHECK: call void @decl_unknown()
  call void @unknown_callee()
  ret void
}

define void @test_inlined_unknown() {
; CHECK-LABEL: define void @test_inlined_unknown(
; CHECK: store i32 7, ptr @X
; CHECK-NEXT: call void @inlines_unknown()
; CHECK-NEXT: store i32 8, ptr @X
  store i32 7, ptr @X
  call void @inlines_unknown()
  store i32 8, ptr @X
  ret void
}

define internal void @indirect_callee(ptr %fp) {
  call void %fp()
  ret void
}

define void @inlines_indirect(ptr %fp) noinline nounwind {
; CHECK-LABEL: define void @inlines_indirect(
; CHECK: call void %fp()
  call void @indirect_callee(ptr %fp)
  ret void
}

define void @test_inlined_indirect(ptr %fp) {
; CHECK-LABEL: define void @test_inlined_indirect(
; CHECK: store i32 9, ptr @X
; CHECK-NEXT: call void @inlines_indirect(ptr %fp)
; CHECK-NEXT: store i32 10, ptr @X
  store i32 9, ptr @X
  call void @inlines_indirect(ptr %fp)
  store i32 10, ptr @X
  ret void
}

define internal void @readnone_callee() {
  call void @decl_readnone()
  ret void
}

define void @inlines_readnone() noinline nounwind {
; CHECK-LABEL: define void @inlines_readnone(
; CHECK: call void @decl_readnone()
  call void @readnone_callee()
  ret void
}

define void @test_inlined_readnone() {
; CHECK-LABEL: define void @test_inlined_readnone(
; CHECK-NOT: store i32 11, ptr @X
; CHECK: call void @inlines_readnone()
; CHECK-NEXT: store i32 12, ptr @X
  store i32 11, ptr @X
  call void @inlines_readnone()
  store i32 12, ptr @X
  ret void
}
