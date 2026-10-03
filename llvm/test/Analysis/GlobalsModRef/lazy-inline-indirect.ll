; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='cgscc(inline),require<globals-aa>,dse' -S | FileCheck %s
;
; After bar() is inlined into foo(), foo() contains an indirect call.
; GlobalsAA must model the current IR instead of a stale call-graph cache,
; so DSE must keep both stores in @main. The nounwind attributes are
; load-bearing: may-throw calls keep stores via unwind paths regardless
; of AA.

@G = internal global i32 0

define internal void @bar(ptr %fp) {
  call void %fp()
  ret void
}

define void @foo(ptr %fp) noinline nounwind {
; CHECK-LABEL: define void @foo(
; CHECK: call void %{{.*}}()
  call void @bar(ptr %fp)
  ret void
}

define void @main(ptr %fp) {
; CHECK-LABEL: define void @main(
; CHECK: store i32 1, ptr @G
; CHECK-NEXT: call void @foo(
; CHECK-NEXT: store i32 2, ptr @G
  store i32 1, ptr @G
  call void @foo(ptr %fp)
  store i32 2, ptr @G
  ret void
}
