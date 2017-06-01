; This test tries to ensure that the inliner successfully invalidates function
; analyses after inlining into the function body.
;
; The strategy for these tests is to compute domtree over all the functions,
; then run the inliner, and then verify the domtree. Then we can arrange the
; inline to disturb the domtree (easy) and detect any stale cached entries in
; the verifier. We do the initial computation both *inside* the CGSCC walk and
; in a pre-step to make sure both work.
;
; RUN: opt < %s -passes='function(require<domtree>),cgscc(inline,function(verify<domtree>))' -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(function(require<domtree>),inline,function(verify<domtree>))' -S | FileCheck %s

; An external function used to control branches.
declare i1 @flag()
; CHECK-LABEL: declare i1 @flag()

; The utility function with interesting control flow that gets inlined below to
; perturb the dominator tree.
define internal void @callee() {
; CHECK-LABEL: @callee
entry:
  %ptr = alloca i8
  %flag = call i1 @flag()
  br i1 %flag, label %then, label %else

then:
  store volatile i8 42, i8* %ptr
  br label %return

else:
  store volatile i8 -42, i8* %ptr
  br label %return

return:
  ret void
}


; The 'test1_' prefixed functions test the basic scenario of inlining
; destroying dominator tree.

define void @test1_caller() {
; CHECK-LABEL: define void @test1_caller()
entry:
  call void @callee()
; CHECK-NOT: @callee
  ret void
; CHECK: ret void
}


; The 'test2_' prefixed functions test the scenario of not inlining preserving
; dominators.

define void @test2_caller() {
; CHECK-LABEL: define void @test2_caller()
entry:
  call void @callee() noinline
; CHECK: call void @callee
  ret void
; CHECK: ret void
}


; The 'test3_' prefixed functions test the scenario of not inlining preserving
; dominators after splitting an SCC into two smaller SCCs.

; The first function gets visited first and we end up inlining everything we
; can into this routine. That splits test3_g into a separate SCC that is enqued
; for later processing.
define void @test3_f() {
; CHECK-LABEL: define void @test3_f()
entry:
  ; Create the first edge in the SCC cycle.
  call void @test3_g()
; CHECK-NOT: @test3_g()
; CHECK: call void @test3_f()

  ; Pull interesting CFG into this function.
  call void @callee()
; CHECK-NOT: call void @callee()

  ret void
; CHECK: ret void
}

; This function ends up split into a separate SCC, which can cause its analyses
; to become stale if the splitting doesn't properly invalidate things. Also, as
; a consequence of being split out, test3_f is too large to inline by the time
; we get here.
define void @test3_g() {
; CHECK-LABEL: define void @test3_g()
entry:
  ; Create the second edge in the SCC cycle.
  call void @test3_f()
; CHECK: call void @test3_f()

  ; Pull interesting CFG into this function.
  call void @callee()
; CHECK-NOT: call void @callee()

  ret void
; CHECK: ret void
}
