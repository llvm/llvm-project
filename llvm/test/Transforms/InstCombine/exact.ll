; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: @sdiv1
; CHECK: sdiv i32 %x, 8
define i32 @sdiv1(i32 %x) {
  %y = sdiv i32 %x, 8
  ret i32 %y
}

; CHECK: @sdiv2
; CHECK: ashr exact i32 %x, 3
define i32 @sdiv2(i32 %x) {
  %y = sdiv exact i32 %x, 8
  ret i32 %y
}

; CHECK: @sdiv3
; CHECK: %y = srem i32 %x, 3
; CHECK: %z = sub i32 %x, %y
; CHECK: ret i32 %z
define i32 @sdiv3(i32 %x) {
  %y = sdiv i32 %x, 3
  %z = mul i32 %y, 3
  ret i32 %z
}

; CHECK: @sdiv4
; CHECK: ret i32 %x
define i32 @sdiv4(i32 %x) {
  %y = sdiv exact i32 %x, 3
  %z = mul i32 %y, 3
  ret i32 %z
}

; CHECK: i32 @sdiv5
; CHECK: %y = srem i32 %x, 3
; CHECK: %z = sub i32 %y, %x
; CHECK: ret i32 %z
define i32 @sdiv5(i32 %x) {
  %y = sdiv i32 %x, 3
  %z = mul i32 %y, -3
  ret i32 %z
}

; CHECK: @sdiv6
; CHECK: %z = sub i32 0, %x
; CHECK: ret i32 %z
define i32 @sdiv6(i32 %x) {
  %y = sdiv exact i32 %x, 3
  %z = mul i32 %y, -3
  ret i32 %z
}

; CHECK: @udiv1
; CHECK: ret i32 %x
define i32 @udiv1(i32 %x, i32 %w) {
  %y = udiv exact i32 %x, %w
  %z = mul i32 %y, %w
  ret i32 %z
}

; CHECK: @udiv2
; CHECK: %z = lshr exact i32 %x, %w
; CHECK: ret i32 %z
define i32 @udiv2(i32 %x, i32 %w) {
  %y = shl i32 1, %w
  %z = udiv exact i32 %x, %y
  ret i32 %z
}

; CHECK: @ashr1
; CHECK: %B = ashr exact i64 %A, 2
; CHECK: ret i64 %B
define i64 @ashr1(i64 %X) nounwind {
  %A = shl i64 %X, 8
  %B = ashr i64 %A, 2   ; X/4
  ret i64 %B
}

; PR9120
; CHECK: @ashr_icmp1
; CHECK: %B = icmp eq i64 %X, 0
; CHECK: ret i1 %B
define i1 @ashr_icmp1(i64 %X) nounwind {
  %A = ashr exact i64 %X, 2   ; X/4
  %B = icmp eq i64 %A, 0
  ret i1 %B
}

; CHECK: @ashr_icmp2
; CHECK: %Z = icmp slt i64 %X, 16
; CHECK: ret i1 %Z
define i1 @ashr_icmp2(i64 %X) nounwind {
 %Y = ashr exact i64 %X, 2  ; x / 4
 %Z = icmp slt i64 %Y, 4    ; x < 16
 ret i1 %Z
}

; PR9998
; Make sure we don't transform the ashr here into an sdiv
; CHECK: @pr9998
; CHECK: = and i32 %V, 1
; CHECK: %Z = icmp ne
; CHECK: ret i1 %Z
define i1 @pr9998(i32 %V) nounwind {
entry:
  %W = shl i32 %V, 31
  %X = ashr exact i32 %W, 31
  %Y = sext i32 %X to i64
  %Z = icmp ugt i64 %Y, 7297771788697658747
  ret i1 %Z
}


; CHECK: @udiv_icmp1
; CHECK: icmp ne i64 %X, 0
define i1 @udiv_icmp1(i64 %X) nounwind {
  %A = udiv exact i64 %X, 5   ; X/5
  %B = icmp ne i64 %A, 0
  ret i1 %B
}

; CHECK: @sdiv_icmp1
; CHECK: icmp eq i64 %X, 0
define i1 @sdiv_icmp1(i64 %X) nounwind {
  %A = sdiv exact i64 %X, 5   ; X/5 == 0 --> x == 0
  %B = icmp eq i64 %A, 0
  ret i1 %B
}

; CHECK: @sdiv_icmp2
; CHECK: icmp eq i64 %X, 5
define i1 @sdiv_icmp2(i64 %X) nounwind {
  %A = sdiv exact i64 %X, 5   ; X/5 == 1 --> x == 5
  %B = icmp eq i64 %A, 1
  ret i1 %B
}

; CHECK: @sdiv_icmp3
; CHECK: icmp eq i64 %X, -5
define i1 @sdiv_icmp3(i64 %X) nounwind {
  %A = sdiv exact i64 %X, 5   ; X/5 == -1 --> x == -5
  %B = icmp eq i64 %A, -1
  ret i1 %B
}

; CHECK: @sdiv_icmp4
; CHECK: icmp eq i64 %X, 0
define i1 @sdiv_icmp4(i64 %X) nounwind {
  %A = sdiv exact i64 %X, -5   ; X/-5 == 0 --> x == 0
  %B = icmp eq i64 %A, 0
  ret i1 %B
}

; CHECK: @sdiv_icmp5
; CHECK: icmp eq i64 %X, -5
define i1 @sdiv_icmp5(i64 %X) nounwind {
  %A = sdiv exact i64 %X, -5   ; X/-5 == 1 --> x == -5
  %B = icmp eq i64 %A, 1
  ret i1 %B
}

; CHECK: @sdiv_icmp6
; CHECK: icmp eq i64 %X, 5
define i1 @sdiv_icmp6(i64 %X) nounwind {
  %A = sdiv exact i64 %X, -5   ; X/-5 == 1 --> x == 5
  %B = icmp eq i64 %A, -1
  ret i1 %B
}

