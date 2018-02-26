; This test contains extremely tricky call graph structures for the inliner to
; handle correctly. They form cycles where the inliner introduces code that is
; immediately or can eventually be transformed back into the original code. And
; each step changes the call graph and so will trigger iteration. This requires
; some out-of-band way to prevent infinitely re-inlining and re-transforming the
; code.
;
; RUN: opt < %s -passes='cgscc(inline,function(sroa,instcombine))' -S | FileCheck %s


; The `test1_*` collection of functions form a directly cycling pattern.

define void @test1_a(i8** %ptr) {
; CHECK-LABEL: define void @test1_a(
entry:
  call void @test1_b(i8* bitcast (void (i8*, i1, i32)* @test1_b to i8*), i1 false, i32 0)
; Inlining and simplifying this call will reliably produce the exact same call,
; over and over again. However, each inlining increments the count, and so we
; expect this test case to stop after one round of inlining with a final
; argument of '1'.
; CHECK-NOT:     call
; CHECK:         call void @test1_b(i8* bitcast (void (i8*, i1, i32)* @test1_b to i8*), i1 false, i32 1)
; CHECK-NOT:     call

  ret void
}

define void @test1_b(i8* %arg, i1 %flag, i32 %inline_count) {
; CHECK-LABEL: define void @test1_b(
entry:
  %a = alloca i8*
  store i8* %arg, i8** %a
; This alloca and store should remain through any optimization.
; CHECK:         %[[A:.*]] = alloca
; CHECK:         store i8* %arg, i8** %[[A]]

  br i1 %flag, label %bb1, label %bb2

bb1:
  call void @test1_a(i8** %a) noinline
  br label %bb2

bb2:
  %cast = bitcast i8** %a to void (i8*, i1, i32)**
  %p = load void (i8*, i1, i32)*, void (i8*, i1, i32)** %cast
  %inline_count_inc = add i32 %inline_count, 1
  call void %p(i8* %arg, i1 %flag, i32 %inline_count_inc)
; And we should continue to load and call indirectly through optimization.
; CHECK:         %[[CAST:.*]] = bitcast i8** %[[A]] to void (i8*, i1, i32)**
; CHECK:         %[[P:.*]] = load void (i8*, i1, i32)*, void (i8*, i1, i32)** %[[CAST]]
; CHECK:         call void %[[P]](

  ret void
}

define void @test2_a(i8** %ptr) {
; CHECK-LABEL: define void @test2_a(
entry:
  call void @test2_b(i8* bitcast (void (i8*, i8*, i1, i32)* @test2_b to i8*), i8* bitcast (void (i8*, i8*, i1, i32)* @test2_c to i8*), i1 false, i32 0)
; Inlining and simplifying this call will reliably produce the exact same call,
; but only after doing two rounds if inlining, first from @test2_b then
; @test2_c. We check the exact number of inlining rounds before we cut off to
; break the cycle by inspecting the last paramater that gets incremented with
; each inlined function body.
; CHECK-NOT:     call
; CHECK:         call void @test2_b(i8* bitcast (void (i8*, i8*, i1, i32)* @test2_b to i8*), i8* bitcast (void (i8*, i8*, i1, i32)* @test2_c to i8*), i1 false, i32 2)
; CHECK-NOT:     call
  ret void
}

define void @test2_b(i8* %arg1, i8* %arg2, i1 %flag, i32 %inline_count) {
; CHECK-LABEL: define void @test2_b(
entry:
  %a = alloca i8*
  store i8* %arg2, i8** %a
; This alloca and store should remain through any optimization.
; CHECK:         %[[A:.*]] = alloca
; CHECK:         store i8* %arg2, i8** %[[A]]

  br i1 %flag, label %bb1, label %bb2

bb1:
  call void @test2_a(i8** %a) noinline
  br label %bb2

bb2:
  %p = load i8*, i8** %a
  %cast = bitcast i8* %p to void (i8*, i8*, i1, i32)*
  %inline_count_inc = add i32 %inline_count, 1
  call void %cast(i8* %arg1, i8* %arg2, i1 %flag, i32 %inline_count_inc)
; And we should continue to load and call indirectly through optimization.
; CHECK:         %[[CAST:.*]] = bitcast i8** %[[A]] to void (i8*, i8*, i1, i32)**
; CHECK:         %[[P:.*]] = load void (i8*, i8*, i1, i32)*, void (i8*, i8*, i1, i32)** %[[CAST]]
; CHECK:         call void %[[P]](

  ret void
}

define void @test2_c(i8* %arg1, i8* %arg2, i1 %flag, i32 %inline_count) {
; CHECK-LABEL: define void @test2_c(
entry:
  %a = alloca i8*
  store i8* %arg1, i8** %a
; This alloca and store should remain through any optimization.
; CHECK:         %[[A:.*]] = alloca
; CHECK:         store i8* %arg1, i8** %[[A]]

  br i1 %flag, label %bb1, label %bb2

bb1:
  call void @test2_a(i8** %a) noinline
  br label %bb2

bb2:
  %p = load i8*, i8** %a
  %cast = bitcast i8* %p to void (i8*, i8*, i1, i32)*
  %inline_count_inc = add i32 %inline_count, 1
  call void %cast(i8* %arg1, i8* %arg2, i1 %flag, i32 %inline_count_inc)
; And we should continue to load and call indirectly through optimization.
; CHECK:         %[[CAST:.*]] = bitcast i8** %[[A]] to void (i8*, i8*, i1, i32)**
; CHECK:         %[[P:.*]] = load void (i8*, i8*, i1, i32)*, void (i8*, i8*, i1, i32)** %[[CAST]]
; CHECK:         call void %[[P]](

  ret void
}
