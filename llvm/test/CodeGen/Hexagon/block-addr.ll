; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: .LJTI
; CHECK-DAG: r[[REG:[0-9]+]] = memw(r{{[0-9]+}}{{ *}}+{{ *}}r{{[0-9]+<<#[0-9]+}})
; CHECK-DAG: jumpr r[[REG]]

define void @main() #0 {
entry:
  %ret = alloca i32, align 4
  br label %while.body

while.body:
  %ret.0.load17 = load volatile i32, i32* %ret, align 4
  switch i32 %ret.0.load17, label %label6 [
    i32 0, label %label0
    i32 1, label %label1
    i32 2, label %label2
    i32 3, label %label3
    i32 4, label %label4
    i32 5, label %label5
  ]

label0:
  %ret.0.load18 = load volatile i32, i32* %ret, align 4
  %inc = add nsw i32 %ret.0.load18, 1
  store volatile i32 %inc, i32* %ret, align 4
  br label %while.body

label1:
  %ret.0.load19 = load volatile i32, i32* %ret, align 4
  %inc2 = add nsw i32 %ret.0.load19, 1
  store volatile i32 %inc2, i32* %ret, align 4
  br label %while.body

label2:
  %ret.0.load20 = load volatile i32, i32* %ret, align 4
  %inc4 = add nsw i32 %ret.0.load20, 1
  store volatile i32 %inc4, i32* %ret, align 4
  br label %while.body

label3:
  %ret.0.load21 = load volatile i32, i32* %ret, align 4
  %inc6 = add nsw i32 %ret.0.load21, 1
  store volatile i32 %inc6, i32* %ret, align 4
  br label %while.body

label4:
  %ret.0.load22 = load volatile i32, i32* %ret, align 4
  %inc8 = add nsw i32 %ret.0.load22, 1
  store volatile i32 %inc8, i32* %ret, align 4
  br label %while.body

label5:
  %ret.0.load23 = load volatile i32, i32* %ret, align 4
  %inc10 = add nsw i32 %ret.0.load23, 1
  store volatile i32 %inc10, i32* %ret, align 4
  br label %while.body

label6:
  store volatile i32 0, i32* %ret, align 4
  br label %while.body
}

attributes #0 = { noreturn nounwind "target-cpu"="hexagonv4" }
