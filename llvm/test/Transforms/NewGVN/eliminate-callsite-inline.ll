; RUN: opt -inline -newgvn -S < %s | FileCheck %s

; CHECK-LABEL: @f2()
; CHECK-NEXT:    ret void
; CHECK-NOT: @f1

define void @f2() {
  call void @f1()
  call void @f1()
  ret void
}

define internal void @f1() #1 {
entry:
  ret void
}

attributes #1 = { noinline nounwind readnone }
