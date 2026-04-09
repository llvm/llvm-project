; RUN: llc < %s -verify-machineinstrs -enable-machine-outliner | FileCheck %s

target triple = "riscv64-unknown-linux-gnu"

declare void @foo(i32, i32, i32, i32) minsize

define void @fentry0(i1 %a) nounwind {
; CHECK-LABEL: fentry0:
; CHECK:       # %bb.1:
; CHECK-NEXT:    call t0, OUTLINED_FUNCTION_[[BB1:[0-9]+]]
; CHECK-NEXT:    call foo
; CHECK-LABEL: .LBB0_2:
; CHECK-NEXT:    tail OUTLINED_FUNCTION_[[BB2:[0-9]+]]
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

define void @fentry1(i1 %a) nounwind {
; CHECK-LABEL: fentry1:
; CHECK:       # %bb.1:
; CHECK-NEXT:    call t0, OUTLINED_FUNCTION_[[BB1:[0-9]+]]
; CHECK-NEXT:    call foo
; CHECK-LABEL: .LBB1_2:
; CHECK-NEXT:    tail OUTLINED_FUNCTION_[[BB2:[0-9]+]]
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

define void @fentry2(i1 %a) nounwind {
; CHECK-LABEL: fentry2:
; CHECK:       # %bb.1:
; CHECK-NEXT:    call t0, OUTLINED_FUNCTION_[[BB1:[0-9]+]]
; CHECK-NEXT:    call foo
; CHECK-LABEL: .LBB2_2:
; CHECK-NEXT:    tail OUTLINED_FUNCTION_[[BB2:[0-9]+]]
entry:
  br i1 %a, label %if.then, label %if.end
if.then:
  call void @foo(i32 1, i32 2, i32 3, i32 4)
  br label %if.end
if.end:
  call void @foo(i32 5, i32 6, i32 7, i32 8)
  ret void
}

; CHECK:       OUTLINED_FUNCTION_[[BB2]]:
; CHECK:       li      a0, 5
; CHECK-NEXT:  li      a1, 6
; CHECK-NEXT:  li      a2, 7
; CHECK-NEXT:  li      a3, 8
; CHECK-NEXT:  call foo

; CHECK:       OUTLINED_FUNCTION_[[BB1]]:
; CHECK:       li      a0, 1
; CHECK-NEXT:  li      a1, 2
; CHECK-NEXT:  li      a2, 3
; CHECK-NEXT:  li      a3, 4
; CHECK-NEXT:  jr      t0
