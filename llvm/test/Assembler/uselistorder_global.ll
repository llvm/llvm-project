; RUN: opt -S -preserve-ll-uselistorder < %s | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: @g = external global i32
; CHECK: define void @func1() {
; CHECK-NOT: uselistorder
; CHECK: }
; CHECK: define void @func2() {
; CHECK-NOT: uselistorder
; CHECK: }
; CHECK: uselistorder i32* @g, { 3, 2, 1, 0 }

@g = external global i32

define void @func1() {
  load i32, i32* @g
  load i32, i32* @g
  ret void
}

define void @func2() {
  load i32, i32* @g
  load i32, i32* @g
  ret void
}

uselistorder i32* @g, { 3, 2, 1, 0 }
