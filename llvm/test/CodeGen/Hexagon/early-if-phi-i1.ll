; RUN: llc -march=hexagon < %s
; REQUIRES: asserts
; Check that the early if-conversion does not predicate block1 (where the
; join block has a phi node of type i1).

define i1 @foo(i32 %x, ptr %p) {
entry:
  %c = icmp sgt i32 %x, 0
  %c1 = icmp sgt i32 %x, 10
  br i1 %c, label %block2, label %block1
block1:
  store i32 1, ptr %p, align 4
  br label %block2
block2:
  %b = phi i1 [ 0, %entry ], [ %c1, %block1 ]
  ret i1 %b
}
