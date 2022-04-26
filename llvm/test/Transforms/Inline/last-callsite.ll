; RUN: opt < %s -passes='cgscc(inline)' -inline-threshold=0 -S | FileCheck %s

; The 'test1_' prefixed functions test the basic 'last callsite' inline
; threshold adjustment where we specifically inline the last call site of an
; internal function regardless of cost.

define internal void @test1_f() {
entry:
  %p = alloca i32
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  ret void
}

; Identical to @test1_f but doesn't get inlined because there is more than one
; call. If this *does* get inlined, the body used both here and in @test1_f
; isn't a good test for different threshold based on the last call.
define internal void @test1_g() {
entry:
  %p = alloca i32
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  ret void
}

define void @test1() {
; CHECK-LABEL: define void @test1()
entry:
  call void @test1_f()
; CHECK-NOT: @test1_f

  call void @test1_g()
  call void @test1_g()
; CHECK: call void @test1_g()
; CHECK: call void @test1_g()

  ret void
}


; The 'test2_' prefixed functions test that we can discover the last callsite
; bonus after having inlined the prior call site. For this to work, we need
; a callsite dependent cost so we have a trivial predicate guarding all the
; cost, and set that in a particular direction.

define internal void @test2_f(i1 %b) {
entry:
  %p = alloca i32
  br i1 %b, label %then, label %exit

then:
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  br label %exit

exit:
  ret void
}

; Identical to @test2_f but doesn't get inlined because there is more than one
; call. If this *does* get inlined, the body used both here and in @test2_f
; isn't a good test for different threshold based on the last call.
define internal void @test2_g(i1 %b) {
entry:
  %p = alloca i32
  br i1 %b, label %then, label %exit

then:
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  br label %exit

exit:
  ret void
}

define void @test2() {
; CHECK-LABEL: define void @test2()
entry:
  ; The first call is trivial to inline due to the argument.
  call void @test2_f(i1 false)
; CHECK-NOT: @test2_f

  ; The second call is too expensive to inline unless we update the number of
  ; calls after inlining the second.
  call void @test2_f(i1 true)
; CHECK-NOT: @test2_f

  ; Check that two calls with the hard predicate remain uninlined.
  call void @test2_g(i1 true)
  call void @test2_g(i1 true)
; CHECK: call void @test2_g(i1 true)
; CHECK: call void @test2_g(i1 true)

  ret void
}


; The 'test3_' prefixed functions are similar to the 'test2_' functions but the
; relative order of the trivial and hard to inline callsites is reversed. This
; checks that the order of calls isn't significant to whether we observe the
; "last callsite" threshold difference because the next-to-last gets inlined.
; FIXME: We don't currently catch this case.

define internal void @test3_f(i1 %b) {
entry:
  %p = alloca i32
  br i1 %b, label %then, label %exit

then:
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  br label %exit

exit:
  ret void
}

; Identical to @test3_f but doesn't get inlined because there is more than one
; call. If this *does* get inlined, the body used both here and in @test3_f
; isn't a good test for different threshold based on the last call.
define internal void @test3_g(i1 %b) {
entry:
  %p = alloca i32
  br i1 %b, label %then, label %exit

then:
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  br label %exit

exit:
  ret void
}

define void @test3() {
; CHECK-LABEL: define void @test3()
entry:
  ; The first call is too expensive to inline unless we update the number of
  ; calls after inlining the second.
  call void @test3_f(i1 true)
; FIXME: We should inline this call without iteration.
; CHECK: call void @test3_f(i1 true)

  ; But the second call is trivial to inline due to the argument.
  call void @test3_f(i1 false)
; CHECK-NOT: @test3_f

  ; Check that two calls with the hard predicate remain uninlined.
  call void @test3_g(i1 true)
  call void @test3_g(i1 true)
; CHECK: call void @test3_g(i1 true)
; CHECK: call void @test3_g(i1 true)

  ret void
}


; The 'test4_' prefixed functions are similar to the 'test2_' prefixed
; functions but include unusual constant expressions that make discovering that
; a function is dead harder.

define internal void @test4_f(i1 %b) {
entry:
  %p = alloca i32
  br i1 %b, label %then, label %exit

then:
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  br label %exit

exit:
  ret void
}

; Identical to @test4_f but doesn't get inlined because there is more than one
; call. If this *does* get inlined, the body used both here and in @test4_f
; isn't a good test for different threshold based on the last call.
define internal void @test4_g(i1 %b) {
entry:
  %p = alloca i32
  br i1 %b, label %then, label %exit

then:
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  store volatile i32 0, i32* %p
  br label %exit

exit:
  ret void
}

define void @test4() {
; CHECK-LABEL: define void @test4()
entry:
  ; The first call is trivial to inline due to the argument. However this
  ; argument also uses the function being called as part of a complex
  ; constant expression. Merely inlining and deleting the call isn't enough to
  ; drop the use count here, we need to GC the dead constant expression as
  ; well.
  call void @test4_f(i1 icmp ne (i64 ptrtoint (void (i1)* @test4_f to i64), i64 ptrtoint(void (i1)* @test4_f to i64)))
; CHECK-NOT: @test4_f

  ; The second call is too expensive to inline unless we update the number of
  ; calls after inlining the second.
  call void @test4_f(i1 true)
; CHECK-NOT: @test4_f

  ; And check that a single call to a function which is used by a complex
  ; constant expression cannot be inlined because the constant expression forms
  ; a second use. If this part starts failing we need to use more complex
  ; constant expressions to reference a particular function with them.
  %sink = alloca i64
  store volatile i64 mul (i64 ptrtoint (void (i1)* @test4_g to i64), i64 ptrtoint(void (i1)* @test4_g to i64)), i64* %sink
  call void @test4_g(i1 true)
; CHECK: store volatile i64 mul (i64 ptrtoint (void (i1)* @test4_g to i64), i64 ptrtoint (void (i1)* @test4_g to i64)), i64* %sink
; CHECK: call void @test4_g(i1 true)

  ret void
}
