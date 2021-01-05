; RUN: opt < %s -analyze -block-freq -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -analyze -lazy-block-freq -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

define void @test1() {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test1':
; CHECK-NEXT: block-frequency-info: test1
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %loop

; CHECK-NEXT: loop: float = 16.5
loop:
  switch i32 undef, label %loop [
    i32 0, label %return
    i32 1, label %return
  ]

; CHECK-NEXT: return: float = 1.0
return:
  ret void
}
