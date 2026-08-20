; RUN: opt < %s -passes="print<cost-model>" -cost-kind=latency \
; RUN:   -mtriple=systemz-unknown -mcpu=z13 -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=Z13
; RUN: opt < %s -passes="print<cost-model>" -cost-kind=latency \
; RUN:   -mtriple=systemz-unknown -mcpu=z15 -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=Z15
; RUN: opt < %s -passes="print<cost-model>" -cost-kind=latency \
; RUN:   -mtriple=systemz-unknown -mcpu=z17 -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=Z17

define i8 @load_i8(ptr %p) {
entry:
  %x = load i8, ptr %p
  ret i8 %x
}

define i16 @load_i16(ptr %p) {
entry:
  %x = load i16, ptr %p
  ret i16 %x
}

define i32 @load_i32(ptr %p) {
entry:
  %x = load i32, ptr %p
  ret i32 %x
}

define i64 @load_i64(ptr %p) {
entry:
  %x = load i64, ptr %p
  ret i64 %x
}

define i128 @load_i128(ptr %p) {
entry:
  %x = load i128, ptr %p
  ret i128 %x
}

define <8 x i32> @load_v8i32(ptr %p) {
entry:
  %x = load <8 x i32>, ptr %p
  ret <8 x i32> %x
}

; Z13: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i8, ptr %p, align 1
; Z13: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i16, ptr %p, align 2
; Z13: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i32, ptr %p, align 4
; Z13: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i64, ptr %p, align 8
; Z13: Cost Model: Found an estimated cost of 8 for instruction:   %x = load <8 x i32>, ptr %p, align 32

; Z15: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i8, ptr %p, align 1
; Z15: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i16, ptr %p, align 2
; Z15: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i32, ptr %p, align 4
; Z15: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i64, ptr %p, align 8
; Z15: Cost Model: Found an estimated cost of 8 for instruction:   %x = load <8 x i32>, ptr %p, align 32

; Z17: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i8, ptr %p, align 1
; Z17: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i16, ptr %p, align 2
; Z17: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i32, ptr %p, align 4
; Z17: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i64, ptr %p, align 8
; Z17: Cost Model: Found an estimated cost of 8 for instruction:   %x = load <8 x i32>, ptr %p, align 32
