; RUN: opt < %s -passes=licm -S | FileCheck %s

; Test moved from sinking.ll, as it tests sinking of a store who alone touches
; a memory location in a loop.
; Store can be sunk out of exit block containing indirectbr instructions after
; D50925. Updated to use an argument instead of undef, due to PR38989.
define void @test12(ptr %ptr, ptr %arg) {
; CHECK-LABEL: @test12
; CHECK: store
; CHECK-NEXT: br label %lab4
  br label %lab4

lab4:
  br label %lab20

lab5:
  br label %lab20

lab6:
  br label %lab4

lab7:
  br i1 false, label %lab8, label %lab13

lab8:
  br i1 true, label %lab13, label %lab10

lab10:
  br label %lab7

lab13:
  ret void

lab20:
  br label %lab21

lab21:
; CHECK: lab21:
; CHECK-NOT: store
; CHECK: br i1 false, label %lab21, label %lab22
  store i32 36127957, ptr %ptr, align 4
  br i1 false, label %lab21, label %lab22

lab22:
; CHECK: lab22:
; CHECK-NOT: store
; CHECK-NEXT: indirectbr ptr %arg 
  indirectbr ptr %arg, [label %lab5, label %lab6, label %lab7]
}

