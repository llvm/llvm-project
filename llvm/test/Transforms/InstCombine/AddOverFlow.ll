; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @oppositesign
; CHECK: add nsw i16 %a, %b
define i16 @oppositesign(i16 %x, i16 %y) {
; %a is negative, %b is positive
  %a = or i16 %x, 32768
  %b = and i16 %y, 32767
  %c = add i16 %a, %b
  ret i16 %c
}

define i16 @zero_sign_bit(i16 %a) {
; CHECK-LABEL: @zero_sign_bit(
; CHECK-NEXT: and
; CHECK-NEXT: add nuw
; CHECK-NEXT: ret
  %1 = and i16 %a, 32767
  %2 = add i16 %1, 512
  ret i16 %2
}

define i16 @zero_sign_bit2(i16 %a, i16 %b) {
; CHECK-LABEL: @zero_sign_bit2(
; CHECK-NEXT: and
; CHECK-NEXT: and
; CHECK-NEXT: add nuw
; CHECK-NEXT: ret
  %1 = and i16 %a, 32767
  %2 = and i16 %b, 32767
  %3 = add i16 %1, %2
  ret i16 %3
}

declare i16 @bounded(i16 %input);
declare i32 @__gxx_personality_v0(...);
!0 = !{i16 0, i16 32768} ; [0, 32767]
!1 = !{i16 0, i16 32769} ; [0, 32768]

define i16 @add_bounded_values(i16 %a, i16 %b) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: @add_bounded_values(
entry:
  %c = call i16 @bounded(i16 %a), !range !0
  %d = invoke i16 @bounded(i16 %b) to label %cont unwind label %lpad, !range !0
cont:
; %c and %d are in [0, 32767]. Therefore, %c + %d doesn't unsigned overflow.
  %e = add i16 %c, %d
; CHECK: add nuw i16 %c, %d
  ret i16 %e
lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  ret i16 42
}

define i16 @add_bounded_values_2(i16 %a, i16 %b) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: @add_bounded_values_2(
entry:
  %c = call i16 @bounded(i16 %a), !range !1
  %d = invoke i16 @bounded(i16 %b) to label %cont unwind label %lpad, !range !1
cont:
; Similar to add_bounded_values, but %c and %d are in [0, 32768]. Therefore,
; %c + %d may unsigned overflow and we cannot add NUW.
  %e = add i16 %c, %d
; CHECK: add i16 %c, %d
  ret i16 %e
lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  ret i16 42
}

; CHECK-LABEL: @ripple_nsw1
; CHECK: add nsw i16 %a, %b
define i16 @ripple_nsw1(i16 %x, i16 %y) {
; %a has at most one bit set
  %a = and i16 %y, 1

; %b has a 0 bit other than the sign bit
  %b = and i16 %x, 49151

  %c = add i16 %a, %b
  ret i16 %c
}

; Like the previous test, but flip %a and %b
; CHECK-LABEL: @ripple_nsw2
; CHECK: add nsw i16 %b, %a
define i16 @ripple_nsw2(i16 %x, i16 %y) {
  %a = and i16 %y, 1
  %b = and i16 %x, 49151
  %c = add i16 %b, %a
  ret i16 %c
}

; CHECK-LABEL: @ripple_nsw3
; CHECK: add nsw i16 %a, %b
define i16 @ripple_nsw3(i16 %x, i16 %y) {
  %a = and i16 %y, 43691
  %b = and i16 %x, 21843
  %c = add i16 %a, %b
  ret i16 %c
}

; Like the previous test, but flip %a and %b
; CHECK-LABEL: @ripple_nsw4
; CHECK: add nsw i16 %b, %a
define i16 @ripple_nsw4(i16 %x, i16 %y) {
  %a = and i16 %y, 43691
  %b = and i16 %x, 21843
  %c = add i16 %b, %a
  ret i16 %c
}

; CHECK-LABEL: @ripple_nsw5
; CHECK: add nsw i16 %a, %b
define i16 @ripple_nsw5(i16 %x, i16 %y) {
  %a = or i16 %y, 43691
  %b = or i16 %x, 54613
  %c = add i16 %a, %b
  ret i16 %c
}

; Like the previous test, but flip %a and %b
; CHECK-LABEL: @ripple_nsw6
; CHECK: add nsw i16 %b, %a
define i16 @ripple_nsw6(i16 %x, i16 %y) {
  %a = or i16 %y, 43691
  %b = or i16 %x, 54613
  %c = add i16 %b, %a
  ret i16 %c
}

; CHECK-LABEL: @ripple_no_nsw1
; CHECK: add i32 %a, %x
define i32 @ripple_no_nsw1(i32 %x, i32 %y) {
; We know nothing about %x
  %a = and i32 %y, 1
  %b = add i32 %a, %x
  ret i32 %b
}

; CHECK-LABEL: @ripple_no_nsw2
; CHECK: add nuw i16 %a, %b
define i16 @ripple_no_nsw2(i16 %x, i16 %y) {
; %a has at most one bit set
  %a = and i16 %y, 1

; %b has a 0 bit, but it is the sign bit
  %b = and i16 %x, 32767

  %c = add i16 %a, %b
  ret i16 %c
}

; CHECK-LABEL: @ripple_no_nsw3
; CHECK: add i16 %a, %b
define i16 @ripple_no_nsw3(i16 %x, i16 %y) {
  %a = and i16 %y, 43691
  %b = and i16 %x, 21845
  %c = add i16 %a, %b
  ret i16 %c
}

; Like the previous test, but flip %a and %b
; CHECK-LABEL: @ripple_no_nsw4
; CHECK: add i16 %b, %a
define i16 @ripple_no_nsw4(i16 %x, i16 %y) {
  %a = and i16 %y, 43691
  %b = and i16 %x, 21845
  %c = add i16 %b, %a
  ret i16 %c
}

; CHECK-LABEL: @ripple_no_nsw5
; CHECK: add i16 %a, %b
define i16 @ripple_no_nsw5(i16 %x, i16 %y) {
  %a = or i16 %y, 43689
  %b = or i16 %x, 54613
  %c = add i16 %a, %b
  ret i16 %c
}

; Like the previous test, but flip %a and %b
; CHECK-LABEL: @ripple_no_nsw6
; CHECK: add i16 %b, %a
define i16 @ripple_no_nsw6(i16 %x, i16 %y) {
  %a = or i16 %y, 43689
  %b = or i16 %x, 54613
  %c = add i16 %b, %a
  ret i16 %c
}
