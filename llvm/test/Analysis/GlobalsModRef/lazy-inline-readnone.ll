; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='cgscc(inline),require<globals-aa>,dse' -S | FileCheck %s
;
; After bar() is inlined into foo(), foo() directly calls the readnone
; declaration @decl_readnone(), which has no memory effects. The first store
; in @main is dead and must still be removed (precision check: the fix must
; not just poison everything gained through inlining). The nounwind
; attributes are load-bearing: may-throw calls keep stores via unwind paths
; regardless of AA.

@G = internal global i32 0

declare void @decl_readnone() readnone nounwind

define internal void @bar() {
  call void @decl_readnone()
  ret void
}

define void @foo() noinline nounwind {
; CHECK-LABEL: define void @foo(
; CHECK: call void @decl_readnone()
  call void @bar()
  ret void
}

define void @main() {
; CHECK-LABEL: define void @main(
; CHECK-NOT: store i32 1, ptr @G
; CHECK: call void @foo()
; CHECK-NEXT: store i32 2, ptr @G
  store i32 1, ptr @G
  call void @foo()
  store i32 2, ptr @G
  ret void
}
