; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m0 %s -verify-machineinstrs -o - | FileCheck %s

define i32 @test1a(i32 %a, i32 %b) {
entry:
  %cmp = icmp ne i32 %a, %b
  %cond = zext i1 %cmp to i32
  ret i32 %cond
; CHECK-LABEL: test1a:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:       subs r0, r0, r1
; CHECK-NEXT:  subs r1, r0, #1
; CHECK-NEXT:  sbcs r0, r1
}

define i32 @test1b(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, %b
  %cond = zext i1 %cmp to i32
  ret i32 %cond
; CHECK-LABEL: test1b:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:       subs    r1, r0, r1
; CHECK-NEXT:  movs    r0, #0
; CHECK-NEXT:  subs    r0, r0, r1
; CHECK-NEXT:  adcs    r0, r1
}

define i32 @test2a(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, %b
  %cond = zext i1 %cmp to i32
  ret i32 %cond
; CHECK-LABEL: test2a:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:       subs    r1, r0, r1
; CHECK-NEXT:  movs    r0, #0
; CHECK-NEXT:  subs    r0, r0, r1
; CHECK-NEXT:  adcs    r0, r1
}

define i32 @test2b(i32 %a, i32 %b) {
entry:
  %cmp = icmp ne i32 %a, %b
  %cond = zext i1 %cmp to i32
  ret i32 %cond
; CHECK-LABEL: test2b:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:       subs    r0, r0, r1
; CHECK-NEXT:  subs    r1, r0, #1
; CHECK-NEXT:  sbcs    r0, r1
}

define i32 @test3a(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, %b
  %cond = select i1 %cmp, i32 0, i32 4
  ret i32 %cond
; CHECK-LABEL: test3a:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:       subs    r0, r0, r1
; CHECK-NEXT:  subs    r1, r0, #1
; CHECK-NEXT:  sbcs    r0, r1
; CHECK-NEXT:  lsls    r0, r0, #2
}

define i32 @test3b(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, %b
  %cond = select i1 %cmp, i32 4, i32 0
  ret i32 %cond
; CHECK-LABEL: test3b:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:      subs	r0, r0, r1
; CHECK-NEXT: movs	r1, #0
; CHECK-NEXT: subs	r1, r1, r0
; CHECK-NEXT: adcs	r1, r0
; CHECK-NEXT: lsls	r0, r1, #2
}

; FIXME: This one hasn't changed actually
; but could look like test3b
define i32 @test4a(i32 %a, i32 %b) {
entry:
  %cmp = icmp ne i32 %a, %b
  %cond = select i1 %cmp, i32 0, i32 4
  ret i32 %cond
; CHECK-LABEL: test4a:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:       mov     r2, r0
; CHECK-NEXT:  movs    r0, #0
; CHECK-NEXT:  movs    r3, #4
; CHECK-NEXT:  cmp     r2, r1
; CHECK-NEXT:  bne     .[[BRANCH:[A-Z0-9_]+]]
; CHECK:       mov     r0, r3
; CHECK:       .[[BRANCH]]:
}

define i32 @test4b(i32 %a, i32 %b) {
entry:
  %cmp = icmp ne i32 %a, %b
  %cond = select i1 %cmp, i32 4, i32 0
  ret i32 %cond
; CHECK-LABEL: test4b:
; CHECK-NOT: b{{(ne)|(eq)}}
; CHECK:       subs  r0, r0, r1
; CHECK-NEXT:  subs  r1, r0, #1
; CHECK-NEXT:  sbcs  r0, r1
; CHECK-NEXT:  lsls  r0, r0, #2
}
