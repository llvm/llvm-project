; REQUIRES: asserts
; RUN: opt -passes=inline -mtriple=aarch64--linux-gnu -S -debug-only=inline-cost < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; FIXME: Once the 'or' or 'and' is simplified the second compare is dead, but
; the inline cost model has already added the cost.

define i1 @outer1(i32 %a) {
  %C = call i1 @inner1(i32 0, i32 %a)
  ret i1 %C
}

; CHECK: Analyzing call of inner1
; CHECK: NumInstructionsSimplified: 3
; CHECK: NumInstructions: 4
define i1 @inner1(i32 %a, i32 %b) {
  %tobool = icmp eq i32 %a, 0         ; Simplifies to true
  %tobool1 = icmp eq i32 %b, 0        ; Should be dead once 'or' is simplified
  %or.cond = or i1 %tobool, %tobool1  ; Simplifies to true
  ret i1 %or.cond                     ; Simplifies to ret i1 true
}

define i1 @outer2(i32 %a) {
  %C = call i1 @inner2(i32 1, i32 %a)
  ret i1 %C
}

; CHECK: Analyzing call of inner2
; CHECK: NumInstructionsSimplified: 3
; CHECK: NumInstructions: 4
define i1 @inner2(i32 %a, i32 %b) {
  %tobool = icmp eq i32 %a, 0          ; Simplifies to false
  %tobool1 = icmp eq i32 %b, 0         ; Should be dead once 'and' is simplified
  %and.cond = and i1 %tobool, %tobool1 ; Simplifies to false
  ret i1 %and.cond                     ; Simplifies to ret i1 false
}


define i32 @outer3(i32 %a) {
  %C = call i32 @inner3(i32 4294967295, i32 %a)
  ret i32 %C
}

; CHECK: Analyzing call of inner3
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 2
define i32 @inner3(i32 %a, i32 %b) {
  %or.cond = or i32 %a, %b         ; Simplifies to 4294967295
  ret i32 %or.cond                 ; Simplifies to ret i32 4294967295
}


define i32 @outer4(i32 %a) {
  %C = call i32 @inner4(i32 0, i32 %a)
  ret i32 %C
}

; CHECK: Analyzing call of inner4
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 2
define i32 @inner4(i32 %a, i32 %b) {
  %and.cond = and i32 %a, %b       ; Simplifies to 0
  ret i32 %and.cond                ; Simplifies to ret i32 0
}

define i1 @outer5(i32 %a) {
  %C = call i1 @inner5(i32 0, i32 %a)
  ret i1 %C
}

; CHECK: Analyzing call of inner5
; CHECK: NumInstructionsSimplified: 4
; CHECK: NumInstructions: 5
define i1 @inner5(i32 %a, i32 %b) {
  %tobool = icmp eq i32 %a, 0         ; Simplifies to true
  %tobool1 = icmp eq i32 %b, 0        ; Should be dead once 'or' is simplified
  %or.cond = or i1 %tobool, %tobool1  ; Simplifies to true
  br i1 %or.cond, label %end, label %isfalse ; Simplifies to br label %end

isfalse:             ; This block is unreachable once inlined
  call void @dead()
  call void @dead()
  call void @dead()
  call void @dead()
  call void @dead()
  br label %end

end:
  ret i1 %or.cond    ; Simplifies to ret i1 true
}

declare void @dead()
