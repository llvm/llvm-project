; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: r0 = abs(r0)
define i32 @f0(i32 %a0) #0 {
  %v0 = ashr i32 %a0, 31
  %v1 = xor i32 %a0, %v0
  %v2 = sub i32 %v1, %v0
  ret i32 %v2
}

; CHECK-LABEL: f1:
; CHECK: r0 = abs(r0)
define i32 @f1(i32 %a0) #0 {
  %v0 = ashr i32 %a0, 31
  %v1 = add i32 %a0, %v0
  %v2 = xor i32 %v0, %v1
  ret i32 %v2
}

; CHECK-LABEL: f2:
; CHECK: r0 = abs(r0)
define i32 @f2(i32 %a0) #0 {
  %v0 = icmp slt i32 %a0, 0
  %v1 = sub nsw i32 0, %a0
  %v2 = select i1 %v0, i32 %v1, i32 %a0
  ret i32 %v2
}

; CHECK-LABEL: f3:
; CHECK: r1:0 = abs(r1:0)
define i64 @f3(i64 %a0) #0 {
  %v0 = ashr i64 %a0, 63
  %v1 = xor i64 %a0, %v0
  %v2 = sub i64 %v1, %v0
  ret i64 %v2
}

; CHECK-LABEL: f4:
; CHECK: r1:0 = abs(r1:0)
define i64 @f4(i64 %a0) #0 {
  %v0 = ashr i64 %a0, 63
  %v1 = add i64 %a0, %v0
  %v2 = xor i64 %v0, %v1
  ret i64 %v2
}

; CHECK-LABEL: f5:
; CHECK: r1:0 = abs(r1:0)
define i64 @f5(i64 %a0) #0 {
  %v0 = icmp slt i64 %a0, 0
  %v1 = sub nsw i64 0, %a0
  %v2 = select i1 %v0, i64 %v1, i64 %a0
  ret i64 %v2
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" }
