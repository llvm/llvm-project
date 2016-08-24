; RUN: opt -aa-pipeline=basic-aa -passes='cgscc(function-attrs)' -S < %s | FileCheck %s --check-prefix=BEFORE
; RUN: opt -aa-pipeline=basic-aa -passes='cgscc(function-attrs,function(gvn))' -S < %s | FileCheck %s --check-prefix=AFTER
;
; Also check that adding an extra CGSCC pass after the function update but
; without requiring the outer manager to iterate doesn't break any invariant.
; RUN: opt -aa-pipeline=basic-aa -passes='cgscc(function-attrs,function(gvn),function-attrs)' -S < %s | FileCheck %s --check-prefix=AFTER2

declare void @readnone() readnone
declare void @unknown()

; The @test1_* functions check that when we refine an indirect call to a direct
; call, even if it doesn't change the call graph structure, we revisit the SCC
; passes to reflect the more precise information.
; FIXME: Currently, this isn't implemented in the new pass manager and so we
; only get this with AFTER2, not with AFTER.

; BEFORE: define void @test1_a() {
; AFTER: define void @test1_a() {
; AFTER2: define void @test1_a() {
define void @test1_a() {
  %fptr = alloca void()*
  store void()* @unknown, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; BEFORE: define void @test1_b() {
; AFTER: define void @test1_b() {
; AFTER2: define void @test1_b() #0 {
define void @test1_b() {
  %fptr = alloca void()*
  store void()* @readnone, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; The @test2_* checks that if we refine an indirect call to a direct call and
; in the process change the very structure of the call graph we also revisit
; that component of the graph and do so in an up-to-date fashion.

; BEFORE: define void @test2_a1() {
; AFTER: define void @test2_a1() {
; AFTER2: define void @test2_a1() {
define void @test2_a1() {
  %fptr = alloca void()*
  store void()* @test2_b2, void()** %fptr
  store void()* @test2_b1, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; BEFORE: define void @test2_b1() {
; AFTER: define void @test2_b1() {
; AFTER2: define void @test2_b1() {
define void @test2_b1() {
  call void @unknown()
  call void @test2_a1()
  ret void
}

; BEFORE: define void @test2_a2() {
; AFTER: define void @test2_a2() #0 {
; AFTER2: define void @test2_a2() #0 {
define void @test2_a2() {
  %fptr = alloca void()*
  store void()* @test2_b1, void()** %fptr
  store void()* @test2_b2, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; BEFORE: define void @test2_b2() {
; AFTER: define void @test2_b2() #0 {
; AFTER2: define void @test2_b2() #0 {
define void @test2_b2() {
  call void @readnone()
  call void @test2_a2()
  ret void
}


; The @test3_* set of functions exercise a case where running function passes
; introduces a new post-order relationship that was not present originally and
; makes sure we walk across the SCCs in that order.

; CHECK: define void @test3_a() {
define void @test3_a() {
  call void @test3_b1()
  call void @test3_b2()
  call void @test3_b3()
  call void @unknown()
  ret void
}

; CHECK: define void @test3_b1() #0 {
define void @test3_b1() {
  %fptr = alloca void()*
  store void()* @test3_a, void()** %fptr
  store void()* @readnone, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; CHECK: define void @test3_b2() #0 {
define void @test3_b2() {
  %fptr = alloca void()*
  store void()* @test3_a, void()** %fptr
  store void()* @test3_b2, void()** %fptr
  store void()* @test3_b3, void()** %fptr
  store void()* @test3_b1, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; CHECK: define void @test3_b3() #0 {
define void @test3_b3() {
  %fptr = alloca void()*
  store void()* @test3_a, void()** %fptr
  store void()* @test3_b2, void()** %fptr
  store void()* @test3_b3, void()** %fptr
  store void()* @test3_b1, void()** %fptr
  %f = load void()*, void()** %fptr
  call void %f()
  ret void
}

; CHECK: attributes #0 = { readnone }
