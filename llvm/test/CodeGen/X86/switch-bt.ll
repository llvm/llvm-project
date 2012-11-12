; RUN: llc -march=x86-64 -asm-verbose=false < %s | FileCheck %s

; This switch should use bit tests, and the third bit test case is just
; testing for one possible value, so it doesn't need a bt.

;      CHECK: movabsq $2305843009482129440, %r
; CHECK-NEXT: btq %rax, %r
; CHECK-NEXT: jae
;     CHECK: movl  $671088640, %e
; CHECK-NEXT: btq %rax, %r
; CHECK-NEXT: jae
;      CHECK: testq %rax, %r
; CHECK-NEXT: j

define void @test(i8* %l) nounwind {
entry:
  %l.addr = alloca i8*, align 8                   ; <i8**> [#uses=2]
  store i8* %l, i8** %l.addr
  %tmp = load i8** %l.addr                        ; <i8*> [#uses=1]
  %tmp1 = load i8* %tmp                           ; <i8> [#uses=1]
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
; CHECK: test2:
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
; CHECK: test3:
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
