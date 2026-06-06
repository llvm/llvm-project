; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; OR of two rotates of %a0(r0).
; CHECK-LABEL: f0:
; CHECK: r[[R00:[0-9]+]] = rol(r0,#7)
; CHECK: r[[R00]] |= rol(r0,#9)
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = shl i32 %a0, 7
  %v1 = lshr i32 %a0, 25
  %v2 = or i32 %v0, %v1
  %v3 = shl i32 %a0, 9
  %v4 = lshr i32 %a0, 23
  %v5 = or i32 %v3, %v4
  %v6 = or i32 %v2, %v5
  ret i32 %v6
}

; OR of two rotates of %a0(r0) with an extra input %a1(r1).
; CHECK-LABEL: f1:
; CHECK: r1 |= asl(r0,#7)
; CHECK: r1 |= rol(r0,#9)
define i32 @f1(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = shl i32 %a0, 7
  %v1 = lshr i32 %a0, 25
  %v2 = or i32 %v0, %a1
  %v3 = shl i32 %a0, 9
  %v4 = lshr i32 %a0, 23
  %v5 = or i32 %v3, %v4
  %v6 = or i32 %v2, %v5
  %v7 = or i32 %v6, %v1
  ret i32 %v6
}

; OR of two rotates of two different inputs: %a0(r0) and %a1(r1).
; CHECK-LABEL: f2:
; CHECK: r[[R20:[0-9]+]] = asl(r0,#11)
; CHECK: r[[R21:[0-9]+]] = lsr(r0,#21)
; CHECK: r[[R22:[0-9]+]] = lsr(r1,#13)
; CHECK: r[[R20]] |= asl(r1,#19)
; CHECK: r[[R20]] |= or(r[[R21]],r[[R22]])
define i32 @f2(i32 %a0, i32 %a1) #0 {
  %v0 = shl i32 %a0, 11
  %v1 = lshr i32 %a0, 21
  %v2 = shl i32 %a1, 19
  %v3 = lshr i32 %a1, 13
  %v4 = or i32 %v0, %v2
  %v5 = or i32 %v1, %v3
  %v6 = or i32 %v4, %v5
  ret i32 %v6
}

; ORs of multiple shifts of the same value with only one pair actually
; matching a rotate.
; CHECK-LABEL: f3:
; CHECK: r[[R30:[0-9]+]] = asl(r0,#3)
; CHECK: r[[R30]] |= asl(r0,#5)
; CHECK: r[[R30]] |= asl(r0,#7)
; CHECK: r[[R30]] |= asl(r0,#13)
; CHECK: r[[R30]] |= asl(r0,#19)
; CHECK: r[[R30]] |= lsr(r0,#2)
; CHECK: r[[R30]] |= lsr(r0,#15)
; CHECK: r[[R30]] |= lsr(r0,#23)
; CHECK: r[[R30]] |= lsr(r0,#25)
; CHECK: r[[R30]] |= lsr(r0,#30)
define i32 @f3(i32 %a0) #0 {
  %v0 = shl i32 %a0, 3
  %v1 = shl i32 %a0, 5
  %v2 = shl i32 %a0, 7      ; rotate
  %v3 = shl i32 %a0, 13
  %v4 = shl i32 %a0, 19
  %v5 = lshr i32 %a0, 2
  %v6 = lshr i32 %a0, 15
  %v7 = lshr i32 %a0, 23
  %v8 = lshr i32 %a0, 25    ; rotate
  %v9 = lshr i32 %a0, 30
  %v10 = or i32  %v0, %v1
  %v11 = or i32 %v10, %v2
  %v12 = or i32 %v11, %v3
  %v13 = or i32 %v12, %v4
  %v14 = or i32 %v13, %v5
  %v15 = or i32 %v14, %v6
  %v16 = or i32 %v15, %v7
  %v17 = or i32 %v16, %v8
  %v18 = or i32 %v17, %v9
  ret i32 %v18
}

attributes #0 = { readnone nounwind "target-cpu"="hexagonv60" "target-features"="-packets" }
