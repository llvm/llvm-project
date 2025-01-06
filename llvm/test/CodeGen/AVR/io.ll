; RUN: llc < %s -mtriple=avr | FileCheck %s

define i8 @read8() {
; CHECK-LABEL: read8
; CHECK: in r24, 8
  %1 = load i8, ptr inttoptr (i16 40 to ptr)
  ret i8 %1
}

define i16 @read16() {
; CHECK-LABEL: read16
; CHECK: in r24, 8
; CHECK: in r25, 9
  %1 = load i16, ptr inttoptr (i16 40 to ptr)
  ret i16 %1
}

define i32 @read32() {
; CHECK-LABEL: read32
; CHECK: in r22, 8
; CHECK: in r23, 9
; CHECK: in r24, 10
; CHECK: in r25, 11
  %1 = load i32, ptr inttoptr (i16 40 to ptr)
  ret i32 %1
}

define i64 @read64() {
; CHECK-LABEL: read64
; CHECK: in r18, 8
; CHECK: in r19, 9
; CHECK: in r20, 10
; CHECK: in r21, 11
; CHECK: in r22, 12
; CHECK: in r23, 13
; CHECK: in r24, 14
; CHECK: in r25, 15
  %1 = load i64, ptr inttoptr (i16 40 to ptr)
  ret i64 %1
}

define void @write8() {
; CHECK-LABEL: write8
; CHECK: out 8
  store i8 22, ptr inttoptr (i16 40 to ptr)
  ret void
}

define void @write16() {
; CHECK-LABEL: write16
; CHECK: out 9
; CHECK: out 8
  store i16 1234, ptr inttoptr (i16 40 to ptr)
  ret void
}

define void @write32() {
; CHECK-LABEL: write32
; CHECK: out 11
; CHECK: out 10
; CHECK: out 9
; CHECK: out 8
  store i32 12345678, ptr inttoptr (i16 40 to ptr)
  ret void
}

define void @write64() {
; CHECK-LABEL: write64
; CHECK: out 15
; CHECK: out 14
; CHECK: out 13
; CHECK: out 12
; CHECK: out 11
; CHECK: out 10
; CHECK: out 9
; CHECK: out 8
  store i64 1234567891234567, ptr inttoptr (i16 40 to ptr)
  ret void
}

define void @sbi8() {
; CHECK-LABEL: sbi8
; CHECK: sbi 8, 5
  %1 = load i8, ptr inttoptr (i16 40 to ptr)
  %or = or i8 %1, 32
  store i8 %or, ptr inttoptr (i16 40 to ptr)
  ret void
}

define void @cbi8() {
; CHECK-LABEL: cbi8
; CHECK: cbi 8, 5
  %1 = load volatile i8, ptr inttoptr (i16 40 to ptr)
  %and = and i8 %1, -33
  store volatile i8 %and, ptr inttoptr (i16 40 to ptr)
  ret void
}
