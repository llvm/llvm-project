; RUN: llc < %s -march=x86 | FileCheck %s

define i32 @t1(i8* %X, i32 %i) {
; CHECK: t1:
; CHECK-NOT: and
; CHECK: movzbl
; CHECK: movl (%{{...}},%{{...}},4),
; CHECK: ret

entry:
  %tmp2 = shl i32 %i, 2
  %tmp4 = and i32 %tmp2, 1020
  %tmp7 = getelementptr i8* %X, i32 %tmp4
  %tmp78 = bitcast i8* %tmp7 to i32*
  %tmp9 = load i32* %tmp78
  ret i32 %tmp9
}

define i32 @t2(i16* %X, i32 %i) {
; CHECK: t2:
; CHECK-NOT: and
; CHECK: movzwl
; CHECK: movl (%{{...}},%{{...}},4),
; CHECK: ret

entry:
  %tmp2 = shl i32 %i, 1
  %tmp4 = and i32 %tmp2, 131070
  %tmp7 = getelementptr i16* %X, i32 %tmp4
  %tmp78 = bitcast i16* %tmp7 to i32*
  %tmp9 = load i32* %tmp78
  ret i32 %tmp9
}

define i32 @t3(i16* %i.ptr, i32* %arr) {
; This case is tricky. The lshr followed by a gep will produce a lshr followed
; by an and to remove the low bits. This can be simplified by doing the lshr by
; a greater constant and using the addressing mode to scale the result back up.
; To make matters worse, because of the two-phase zext of %i and their reuse in
; the function, the DAG can get confusing trying to re-use both of them and
; prevent easy analysis of the mask in order to match this.
; CHECK: t3:
; CHECK-NOT: and
; CHECK: shrl
; CHECK: addl (%{{...}},%{{...}},4),
; CHECK: ret

entry:
  %i = load i16* %i.ptr
  %i.zext = zext i16 %i to i32
  %index = lshr i32 %i.zext, 11
  %val.ptr = getelementptr inbounds i32* %arr, i32 %index
  %val = load i32* %val.ptr
  %sum = add i32 %val, %i.zext
  ret i32 %sum
}

define i32 @t4(i16* %i.ptr, i32* %arr) {
; A version of @t3 that has more zero extends and more re-use of intermediate
; values. This exercise slightly different bits of canonicalization.
; CHECK: t4:
; CHECK-NOT: and
; CHECK: shrl
; CHECK: addl (%{{...}},%{{...}},4),
; CHECK: ret

entry:
  %i = load i16* %i.ptr
  %i.zext = zext i16 %i to i32
  %index = lshr i32 %i.zext, 11
  %index.zext = zext i32 %index to i64
  %val.ptr = getelementptr inbounds i32* %arr, i64 %index.zext
  %val = load i32* %val.ptr
  %sum.1 = add i32 %val, %i.zext
  %sum.2 = add i32 %sum.1, %index
  ret i32 %sum.2
}
