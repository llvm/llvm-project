; RUN: opt -analyze -branch-prob < %s -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

declare void @bar() cold

; Both 'l1' and 'r1' has one edge leading to 'cold' and another one to
; 'unreachable' blocks. Check that 'cold' paths are preferred. Also ensure both
; paths from 'entry' block are equal.
define void @test1(i32 %0) {
;CHECK: edge entry -> l1 probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge entry -> r1 probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge l1 -> cold probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge l1 -> unreached probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge r1 -> unreached probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge r1 -> cold probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

entry:
  br i1 undef, label %l1, label %r1

l1:
  br i1 undef, label %cold, label %unreached

r1:
  br i1 undef, label %unreached, label %cold

unreached:
  unreachable

cold:
  call void @bar()
  ret void
}

; Both edges of 'l1' leads to 'cold' blocks while one edge of 'r1' leads to
; 'unreachable' block. Check that 'l1' has 50/50 while 'r1' has 0/100
; distributuion. Also ensure both paths from 'entry' block are equal.
define void @test2(i32 %0) {
;CHECK: edge entry -> l1 probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge entry -> r1 probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge l1 -> cold probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge l1 -> cold2 probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge r1 -> unreached probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge r1 -> cold probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

entry:
  br i1 undef, label %l1, label %r1

l1:
  br i1 undef, label %cold, label %cold2

r1:
  br i1 undef, label %unreached, label %cold

unreached:
  unreachable

cold:
  call void @bar()
  ret void

cold2:
  call void @bar()
  ret void
}

; Both edges of 'r1' leads to 'unreachable' blocks while one edge of 'l1' leads to
; 'cold' block. Ensure that path leading to 'cold' block is preferred.
define void @test3(i32 %0) {
;CHECK: edge entry -> l1 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge entry -> r1 probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge l1 -> cold probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge l1 -> unreached probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge r1 -> unreached probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge r1 -> unreached2 probability is 0x40000000 / 0x80000000 = 50.00%

entry:
  br i1 undef, label %l1, label %r1

l1:
  br i1 undef, label %cold, label %unreached

r1:
  br i1 undef, label %unreached, label %unreached2

unreached:
  unreachable

unreached2:
  unreachable

cold:
  call void @bar()
  ret void
}

; Left edge of 'entry' leads to 'cold' block while right edge is 'normal' continuation.
; Check that we able to propagate 'cold' weight to 'entry' block. Also ensure
; both edges from 'l1' are equally likely.
define void @test4(i32 %0) {
;CHECK: edge entry -> l1 probability is 0x078780e3 / 0x80000000 = 5.88%
;CHECK: edge entry -> r1 probability is 0x78787f1d / 0x80000000 = 94.12% [HOT edge]
;CHECK: edge l1 -> l2 probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge l1 -> r2 probability is 0x40000000 / 0x80000000 = 50.00%
;CHECK: edge l2 -> to.cold probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge r2 -> to.cold probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge to.cold -> cold probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

entry:
  br i1 undef, label %l1, label %r1

l1:
  br i1 undef, label %l2, label %r2

l2:
  br label %to.cold

r2:
  br label %to.cold

to.cold:
 br label %cold

r1:
 ret void

cold:
  call void @bar()
  ret void
}

; Check that most likely path from 'entry' to 'l2' through 'r1' is preferred.
define void @test5(i32 %0) {
;CHECK: edge entry -> cold probability is 0x078780e3 / 0x80000000 = 5.88%
;CHECK: edge entry -> r1 probability is 0x78787f1d / 0x80000000 = 94.12% [HOT edge]
;CHECK: edge cold -> l2 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge r1 -> l2 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge r1 -> unreached probability is 0x00000000 / 0x80000000 = 0.00%

entry:
  br i1 undef, label %cold, label %r1

cold:
  call void @bar()
  br label %l2

r1:
  br i1 undef, label %l2, label %unreached

l2:
  ret void

unreached:
  unreachable
}
