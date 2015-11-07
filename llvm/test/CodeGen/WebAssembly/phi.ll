; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that phis are lowered.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Basic phi triangle.

; CHECK-LABEL: test0:
; CHECK: get_local push, 0{{$}}
; CHECK: set_local [[REG:.*]], pop
; CHECK: div_s push, (get_local [[REG]]), {{.*}}
; CHECK: set_local [[REG]], pop
; CHECK: return (get_local [[REG]])
define i32 @test0(i32 %p) {
entry:
  %t = icmp slt i32 %p, 0
  br i1 %t, label %true, label %done
true:
  %a = sdiv i32 %p, 3
  br label %done
done:
  %s = phi i32 [ %a, %true ], [ %p, %entry ]
  ret i32 %s
}

; Swap phis.

; CHECK-LABEL: test1:
; CHECK: BB1_1:
; CHECK: get_local push, [[REG1:.*]]
; CHECK: set_local [[REG0:.*]], pop
; CHECK: get_local push, [[REG2:.*]]
; CHECK: set_local [[REG1]], pop
; CHECK: [[REG0]]
; CHECK: set_local [[REG2]], pop
define i32 @test1(i32 %n) {
entry:
  br label %loop

loop:
  %a = phi i32 [ 0, %entry ], [ %b, %loop ]
  %b = phi i32 [ 1, %entry ], [ %a, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  %i.next = add i32 %i, 1
  %t = icmp slt i32 %i.next, %n
  br i1 %t, label %loop, label %exit

exit:
  ret i32 %a
}
