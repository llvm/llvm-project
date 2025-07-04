; RUN: opt -expand-constant-exprs %s.bc -S | FileCheck %s
@foo = external constant i32

define i32 @test(i32 %arg) {
entry:
  switch i32 %arg, label %cont [
    i32 1, label %cont
    i32 2, label %nonconst
  ]

nonconst:
  %cmp = icmp ne i32 %arg, 2
  br i1 %cmp, label %cont, label %cont

; CHECK-LABEL: phi.constexpr:
; CHECK-NEXT:    %constexpr = ptrtoint ptr @foo to i32
; CHECK-NEXT:    %constexpr1 = or i32 %constexpr, 5
; CHECK-NEXT:    br label %cont


; CHECK-LABEL: cont:
; CHECK-NEXT:    %res = phi i32 [ %constexpr1, %phi.constexpr ], [ 1, %nonconst ], [ 1, %nonconst ]
; CHECK-NEXT:    ret i32 %res
cont:
  %res = phi i32 [or (i32 5, i32 ptrtoint (ptr @foo to i32)), %entry],
                 [or (i32 5, i32 ptrtoint (ptr @foo to i32)), %entry],
                 [1, %nonconst],
                 [1, %nonconst]
  ret i32 %res
}
