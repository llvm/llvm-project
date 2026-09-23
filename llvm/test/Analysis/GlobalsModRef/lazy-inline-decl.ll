; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='cgscc(inline),require<globals-aa>,dse' -S | FileCheck %s
;
; After bar() is inlined into foo(), foo() directly calls the unknown
; declaration @qux(). GlobalsAA must model the current IR instead of a stale
; call-graph cache, so DSE must keep both stores in @main. The nounwind
; attributes are load-bearing: may-throw calls keep stores via unwind paths
; regardless of AA.

@g = internal global i32 0

declare void @qux()
declare void @opaque(ptr)

define void @g_escape() {
  call void @opaque(ptr @g)
  ret void
}

define internal void @bar() {
  call void @qux()
  ret void
}

define void @foo() noinline nounwind {
; CHECK-LABEL: define void @foo(
; CHECK: call void @qux()
  call void @bar()
  ret void
}

define void @main() {
; CHECK-LABEL: define void @main(
; CHECK: store i32 1, ptr @g
; CHECK-NEXT: call void @foo()
; CHECK-NEXT: store i32 2, ptr @g
  store i32 1, ptr @g
  call void @foo()
  store i32 2, ptr @g
  ret void
}
