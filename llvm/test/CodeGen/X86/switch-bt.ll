; RUN: llc -mtriple=x86_64-- -asm-verbose=false < %s -jump-table-density=40 -switch-peel-threshold=101 | FileCheck %s

; This switch should use bit tests, and the third bit test case is just
; testing for one possible value, so it doesn't need a bt.

;      CHECK: movabsq $2305843009482129440, %r
; CHECK-NEXT: btq %rax, %r
; CHECK-NEXT: jb
;     CHECK: movl  $671088640, %e
; CHECK-NEXT: btq %rax, %r
; CHECK-NEXT: jae
;      CHECK: testq %rax, %r
; CHECK-NEXT: j

define void @test(ptr %l) nounwind {
entry:
  %l.addr = alloca ptr, align 8                   ; <ptr> [#uses=2]
  store ptr %l, ptr %l.addr
  %tmp = load ptr, ptr %l.addr                        ; <ptr> [#uses=1]
  %tmp1 = load i8, ptr %tmp                           ; <i8> [#uses=1]
  %conv = sext i8 %tmp1 to i32                    ; <i32> [#uses=1]
  switch i32 %conv, label %sw.default [
    i32 62, label %sw.bb
    i32 60, label %sw.bb
    i32 38, label %sw.bb2
    i32 94, label %sw.bb2
    i32 61, label %sw.bb2
    i32 33, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry, %entry
  call void @foo(i32 0)
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry, %entry, %entry
  call void @foo(i32 1)
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  call void @foo(i32 3)
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  call void @foo(i32 97)
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb4, %sw.bb2, %sw.bb
  ret void
}

declare void @foo(i32)

; Don't zero extend the test operands to pointer type if it can be avoided.
; rdar://8781238
define void @test2(i32 %x) nounwind ssp {
; CHECK-LABEL: test2:
; CHECK: cmpl $6
; CHECK: ja

; CHECK-NEXT: movl $91
; CHECK-NOT: movl
; CHECK-NEXT: btl
; CHECK-NEXT: jae
entry:
  switch i32 %x, label %if.end [
    i32 6, label %if.then
    i32 4, label %if.then
    i32 3, label %if.then
    i32 1, label %if.then
    i32 0, label %if.then
  ]

if.then:                                          ; preds = %entry, %entry, %entry, %entry, %entry
  tail call void @bar() nounwind
  ret void

if.end:                                           ; preds = %entry
  ret void
}

declare void @bar()

define void @test3(i32 %x) nounwind {
; CHECK-LABEL: test3:
; CHECK: cmpl $5
; CHECK: ja
; CHECK: cmpl $4
; CHECK: je
  switch i32 %x, label %if.end [
    i32 0, label %if.then
    i32 1, label %if.then
    i32 2, label %if.then
    i32 3, label %if.then
    i32 5, label %if.then
  ]
if.then:
  tail call void @bar() nounwind
  ret void
if.end:
  ret void
}

; Ensure that optimizing for jump tables doesn't needlessly deteriorate the
; created binary tree search. See PR22262.
define void @test4(i32 %x, ptr %y) {
; CHECK-LABEL: test4:

entry:
  switch i32 %x, label %sw.default [
    i32 10, label %sw.bb
    i32 20, label %sw.bb1
    i32 30, label %sw.bb2
    i32 40, label %sw.bb3
    i32 50, label %sw.bb4
    i32 60, label %sw.bb5
  ]
sw.bb:
  store i32 1, ptr %y
  br label %sw.epilog
sw.bb1:
  store i32 2, ptr %y
  br label %sw.epilog
sw.bb2:
  store i32 3, ptr %y
  br label %sw.epilog
sw.bb3:
  store i32 4, ptr %y
  br label %sw.epilog
sw.bb4:
  store i32 5, ptr %y
  br label %sw.epilog
sw.bb5:
  store i32 6, ptr %y
  br label %sw.epilog
sw.default:
  store i32 7, ptr %y
  br label %sw.epilog
sw.epilog:
  ret void

; The balanced binary switch here would start with a comparison against 39, but
; it is currently starting with 29 because of the density-sum heuristic.
; CHECK: cmpl $39
; CHECK: jg
; CHECK: cmpl $10
; CHECK: je
; CHECK: cmpl $20
; CHECK: je
; CHECK: cmpl $30
; CHECK: jne
; CHECK: cmpl $40
; CHECK: je
; CHECK: cmpl $50
; CHECK: je
; CHECK: cmpl $60
; CHECK: jne
}


; Omit the range check when the default case is unreachable, see PR43129.
declare void @g(i32)
define void @test5(i32 %x) {

; CHECK-LABEL: test5
; CHECK-NOT: cmp

; 73 = 2^0 + 2^3 + 2^6
; CHECK:      movl $73
; CHECK-NEXT: btl
; CHECK-NEXT: jb

; 146 = 2^1 + 2^4 + 2^7
; CHECK-NEXT: movl $146
; CHECK-NEXT: btl
; CHECK-NEXT: jae


entry:
  switch i32 %x, label %return [
    i32 0, label %bb0
    i32 3, label %bb0
    i32 6, label %bb0

    i32 1, label %bb1
    i32 4, label %bb1
    i32 7, label %bb1

    i32 2, label %bb2
    i32 5, label %bb2
    i32 8, label %bb2
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
return: unreachable
}
