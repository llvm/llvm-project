; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s
; PR1614

; CHECK: -->  (%a smax %b)
; CHECK: -->  (%a smax %b smax %c)
; CHECK-NOT: smax

define i32 @x(i32 %a, i32 %b, i32 %c) {
  %A = icmp sgt i32 %a, %b
  %B = select i1 %A, i32 %a, i32 %b
  %C = icmp sle i32 %c, %B
  %D = select i1 %C, i32 %B, i32 %c
  ret i32 %D
}
