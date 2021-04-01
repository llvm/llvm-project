; RUN: opt < %s -aa-pipeline=basic-aa -passes='cgscc(function-attrs,inline)' -S | FileCheck %s
; This test runs the inliner and the function attribute deduction. It ensures
; that when the inliner mutates the call graph it correctly updates the CGSCC
; iteration so that we can compute refined function attributes. In this way it
; is leveraging function attribute computation to observe correct call graph
; updates.

; Boring unknown external function call.
; CHECK: declare void @unknown()
declare void @unknown()

; Sanity check: this should get annotated as readnone.
; CHECK: Function Attrs: nounwind readnone
; CHECK-NEXT: declare void @readnone()
declare void @readnone() readnone nounwind

; The 'test1_' prefixed functions are designed to trigger forming a new direct
; call in the inlined body of the function. After that, we form a new SCC and
; using that can deduce precise function attrs.

; This function should no longer exist.
; CHECK-NOT: @test1_f()
define internal void @test1_f(void()* %p) {
entry:
  call void %p()
  ret void
}

; This function should have had 'readnone' deduced for its SCC.
; CHECK: Function Attrs: noinline nosync nounwind readnone
; CHECK-NEXT: define void @test1_g()
define void @test1_g() noinline {
entry:
  call void @test1_f(void()* @test1_h)
  ret void
}

; This function should have had 'readnone' deduced for its SCC.
; CHECK: Function Attrs: noinline nosync nounwind readnone
; CHECK-NEXT: define void @test1_h()
define void @test1_h() noinline {
entry:
  call void @test1_g()
  call void @readnone()
  ret void
}


; The 'test2_' prefixed functions are designed to trigger forming a new direct
; call due to RAUW-ing the returned value of a called function into the caller.
; This too should form a new SCC which can then be reasoned about to compute
; precise function attrs.

; This function should no longer exist.
; CHECK-NOT: @test2_f()
define internal void()* @test2_f() {
entry:
  ret void()* @test2_h
}

; This function should have had 'readnone' deduced for its SCC.
; CHECK: Function Attrs: noinline nosync nounwind readnone
; CHECK-NEXT: define void @test2_g()
define void @test2_g() noinline {
entry:
  %p = call void()* @test2_f()
  call void %p()
  ret void
}

; This function should have had 'readnone' deduced for its SCC.
; CHECK: Function Attrs: noinline nosync nounwind readnone
; CHECK-NEXT: define void @test2_h()
define void @test2_h() noinline {
entry:
  call void @test2_g()
  call void @readnone()
  ret void
}


; The 'test3_' prefixed functions are designed to inline in a way that causes
; call sites to become trivially dead during the middle of inlining callsites of
; a single function to make sure that the inliner does not get confused by this
; pattern.

; CHECK-NOT: @test3_maybe_unknown(
define internal void @test3_maybe_unknown(i1 %b) {
entry:
  br i1 %b, label %then, label %exit

then:
  call void @unknown()
  br label %exit

exit:
  ret void
}

; CHECK-NOT: @test3_f(
define internal i1 @test3_f() {
entry:
  ret i1 false
}

; CHECK-NOT: @test3_g(
define internal i1 @test3_g(i1 %b) {
entry:
  br i1 %b, label %then1, label %if2

then1:
  call void @test3_maybe_unknown(i1 true)
  br label %if2

if2:
  %f = call i1 @test3_f()
  br i1 %f, label %then2, label %exit

then2:
  call void @test3_maybe_unknown(i1 true)
  br label %exit

exit:
  ret i1 false
}

; FIXME: Currently the inliner doesn't successfully mark this as readnone
; because while it simplifies trivially dead CFGs when inlining callees it
; doesn't simplify the caller's trivially dead CFG and so we end with a dead
; block calling @unknown.
; CHECK-NOT: Function Attrs: readnone
; CHECK: define void @test3_h()
define void @test3_h() {
entry:
  %g = call i1 @test3_g(i1 false)
  br i1 %g, label %then, label %exit

then:
  call void @test3_maybe_unknown(i1 true)
  br label %exit

exit:
  call void @test3_maybe_unknown(i1 false)
  ret void
}


; The 'test4_' prefixed functions are designed to trigger forming a new direct
; call in the inlined body of the function similar to 'test1_'. However, after
; that we continue to inline another edge of the graph forcing us to do a more
; interesting call graph update for the new call edge. Eventually, we still
; form a new SCC and should use that can deduce precise function attrs.

; This function should have had 'readnone' deduced for its SCC.
; CHECK: Function Attrs: noinline nosync nounwind readnone
; CHECK-NEXT: define void @test4_f1()
define void @test4_f1() noinline {
entry:
  call void @test4_h()
  ret void
}

; CHECK-NOT: @test4_f2
define internal void @test4_f2() {
entry:
  call void @test4_f1()
  ret void
}

; CHECK-NOT: @test4_g
define internal void @test4_g(void()* %p) {
entry:
  call void %p()
  ret void
}

; This function should have had 'readnone' deduced for its SCC.
; CHECK: Function Attrs: noinline nosync nounwind readnone
; CHECK-NEXT: define void @test4_h()
define void @test4_h() noinline {
entry:
  call void @test4_g(void()* @test4_f2)
  ret void
}
