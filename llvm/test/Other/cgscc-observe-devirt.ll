; Make sure that even without some external devirtualization iteration tool,
; the CGSCC pass manager correctly observes and re-visits SCCs that change
; structure due to devirtualization. We trigger devirtualization here with GVN
; which forwards a store through a load and to an indirect call.
;
; RUN: opt -aa-pipeline=basic-aa -passes='cgscc(function-attrs)' -S < %s | FileCheck %s --check-prefix=BEFORE
; RUN: opt -aa-pipeline=basic-aa -passes='cgscc(function-attrs,function(gvn))' -S < %s | FileCheck %s --check-prefix=AFTER
;
; Also check that adding an extra CGSCC pass after the function update but
; without requiring the outer manager to iterate doesn't break any invariant.
; RUN: opt -aa-pipeline=basic-aa -passes='cgscc(function-attrs,function(gvn),function-attrs)' -S < %s | FileCheck %s --check-prefix=AFTER

declare void @readnone() nofree nosync readnone
declare void @unknown()

; The @test1_* checks that if we refine an indirect call to a direct call and
; in the process change the very structure of the call graph we also revisit
; that component of the graph and do so in an up-to-date fashion.

; BEFORE: define void @test1_a1() {
; AFTER: define void @test1_a1() {
define void @test1_a1() {
  %fptr = alloca void()*
  store void()* @test1_b2, void()** %fptr
  store void()* @test1_b1, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; BEFORE: define void @test1_b1() {
; AFTER: define void @test1_b1() {
define void @test1_b1() {
  call void @unknown()
  call void @test1_a1()
  ret void
}

; BEFORE: define void @test1_a2() {
; AFTER: define void @test1_a2() #0 {
define void @test1_a2() {
  %fptr = alloca void()*
  store void()* @test1_b1, void()** %fptr
  store void()* @test1_b2, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; BEFORE: define void @test1_b2() {
; AFTER: define void @test1_b2() #0 {
define void @test1_b2() {
  call void @readnone()
  call void @test1_a2()
  ret void
}


; The @test2_* set of functions exercise a case where running function passes
; introduces a new post-order relationship that was not present originally and
; makes sure we walk across the SCCs in that order.

; CHECK: define void @test2_a() {
define void @test2_a() {
  call void @test2_b1()
  call void @test2_b2()
  call void @test2_b3()
  call void @unknown()
  ret void
}

; CHECK: define void @test2_b1() #0 {
define void @test2_b1() {
  %fptr = alloca void()*
  store void()* @test2_a, void()** %fptr
  store void()* @readnone, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; CHECK: define void @test2_b2() #0 {
define void @test2_b2() {
  %fptr = alloca void()*
  store void()* @test2_a, void()** %fptr
  store void()* @test2_b2, void()** %fptr
  store void()* @test2_b3, void()** %fptr
  store void()* @test2_b1, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; CHECK: define void @test2_b3() #0 {
define void @test2_b3() {
  %fptr = alloca void()*
  store void()* @test2_a, void()** %fptr
  store void()* @test2_b2, void()** %fptr
  store void()* @test2_b3, void()** %fptr
  store void()* @test2_b1, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; CHECK: attributes #0 = { nofree nosync readnone }
