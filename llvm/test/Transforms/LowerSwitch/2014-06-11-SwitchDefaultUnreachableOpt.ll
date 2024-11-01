; RUN: opt < %s -passes=lowerswitch -S | FileCheck %s
;
; The switch is lowered with a single icmp.
; CHECK: icmp
; CHECK-NOT: icmp
;
;int foo(int a) {
;
;  switch (a) {
;  case 0:
;    return 10;
;  case 1:
;    return 3;
;  default:
;    __builtin_unreachable();
;  }
;
;}

define i32 @foo(i32 %a) {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 %a, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  switch i32 %3, label %6 [
    i32 0, label %4
    i32 1, label %5
  ]

; <label>:4 
  store i32 10, i32* %1
  br label %7

; <label>:5
  store i32 3, i32* %1
  br label %7

; <label>:6
  unreachable

; <label>:7
  %8 = load i32, i32* %1
  ret i32 %8
}
