; RUN: opt < %s -disable-output -passes='print<domtree>' 2>&1 | FileCheck %s

define void @test1(i1 %arg) {
; CHECK-LABEL: DominatorTree for function: test1
; CHECK:      [1] %entry
; CHECK-NEXT:   [2] %a
; CHECK-NEXT:   [2] %c
; CHECK-NEXT:     [3] %d
; CHECK-NEXT:     [3] %e
; CHECK-NEXT:   [2] %b

entry:
  br i1 %arg, label %a, label %b

a:
  br label %c

b:
  br label %c

c:
  br i1 %arg, label %d, label %e

d:
  ret void

e:
  ret void
}

define void @test2(i1 %arg) {
; CHECK-LABEL: DominatorTree for function: test2
; CHECK:      [1] %entry
; CHECK-NEXT:   [2] %a
; CHECK-NEXT:     [3] %b
; CHECK-NEXT:       [4] %c
; CHECK-NEXT:         [5] %d
; CHECK-NEXT:         [5] %ret

entry:
  br label %a

a:
  br label %b

b:
  br i1 %arg, label %a, label %c

c:
  br i1 %arg, label %d, label %ret

d:
  br i1 %arg, label %a, label %ret

ret:
  ret void
}
