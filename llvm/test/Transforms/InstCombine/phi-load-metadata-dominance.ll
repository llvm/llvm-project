; RUN: opt -instcombine -S < %s | FileCheck %s

declare void @bar()
declare void @baz()

; Check that nonnull metadata is from non-dominating loads is not propagated.
; CHECK-LABEL: cont:
; CHECK-NOT: !nonnull
define i32* @test_combine_metadata_dominance(i1 %c, i32** dereferenceable(8) %p1, i32** dereferenceable(8) %p2) {
  br i1 %c, label %t, label %f
t:
  call void @bar()
  %v1 = load i32*, i32** %p1, align 8, !nonnull !0
  br label %cont

f:
  call void @baz()
  %v2 = load i32*, i32** %p2, align 8
  br label %cont

cont:
  %res = phi i32* [ %v1, %t ], [ %v2, %f ]
  ret i32* %res
}

!0 = !{}
