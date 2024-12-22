; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that the compiler does not generate an invalid packet with three
; instructions that each requires slot 2 or 3. The specification for
; PS_call_nr was incorrect, which allowed that instrution to go in any slot.

; CHECK: extractu
; CHECK: extractu
; CHECK: {
; CHECK: call

%s.0 = type <{ ptr, ptr, i16, i8, i8, i8 }>

@g0 = external constant %s.0, section ".rodata.trace", align 1

define void @f0() local_unnamed_addr {
b0:
  %v0 = load i32, ptr undef, align 4
  %v1 = trunc i32 %v0 to i2
  switch i2 %v1, label %b4 [
    i2 1, label %b1
    i2 -1, label %b2
    i2 -2, label %b2
    i2 0, label %b3
  ]

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0, %b0
  %v2 = load i32, ptr undef, align 4
  %v3 = lshr i32 %v2, 14
  %v4 = and i32 %v3, 2047
  %v5 = lshr i32 %v2, 3
  %v6 = and i32 %v5, 2047
  tail call void @f1(ptr nonnull @g0, i32 %v6, i32 %v4, i32 0, i32 0)
  unreachable

b3:                                               ; preds = %b0
  ret void

b4:                                               ; preds = %b0
  unreachable
}

declare void @f1(ptr, i32, i32, i32, i32) local_unnamed_addr
