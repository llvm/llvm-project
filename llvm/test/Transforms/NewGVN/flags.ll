; RUN: opt -passes=newgvn -S < %s | FileCheck %s

declare void @use(i1)

define void @test1(float %x, float %y) {
entry:
  %cmp1 = fcmp nnan oeq float %y, %x
  %cmp2 = fcmp oeq float %x, %y
  call void @use(i1 %cmp1)
  call void @use(i1 %cmp2)
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK: %[[cmp:.*]] = fcmp oeq float %y, %x
; CHECK-NEXT: call void @use(i1 %[[cmp]])
; CHECK-NEXT: call void @use(i1 %[[cmp]])
; CHECK-NEXT: ret void
