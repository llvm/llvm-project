; RUN: opt < %s -passes="print<cost-model>" -cost-kind=latency \
; RUN:   -mtriple=systemz-unknown -mcpu=z13 -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=Z13
; RUN: opt < %s -passes="print<cost-model>" -cost-kind=latency \
; RUN:   -mtriple=systemz-unknown -mcpu=z15 -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=Z15
; RUN: opt < %s -passes="print<cost-model>" -cost-kind=latency \
; RUN:   -mtriple=systemz-unknown -mcpu=z17 -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=Z17

define i64 @load_i64(ptr %p) {
entry:
  %x = load i64, ptr %p
  ret i64 %x
}

; Z13: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i64, ptr %p, align 8
; Z15: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i64, ptr %p, align 8
; Z17: Cost Model: Found an estimated cost of 4 for instruction:   %x = load i64, ptr %p, align 8
