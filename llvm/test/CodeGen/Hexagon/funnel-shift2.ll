; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; This patch corrects the order of operands in the pattern that lowers funnel
; shift-left.

; CHECK: r[[R17:[0-9]+]]:[[R16:[0-9]+]] = combine
; CHECK: call parity
; CHECK: r[[R1:[0-9]+]]:[[R0:[0-9]+]] = asl(r[[R1]]:[[R0]],#63)
; CHECK: r[[R1]]:[[R0]] |= lsr(r[[R17]]:[[R16]],#1)

target triple = "hexagon-unknown-unknown-elf"

define dso_local i64 @fshl(i64 %x, i64 %y) {
entry:
  %x.addr = alloca i64, align 8
  %y.addr = alloca i64, align 8
  store i64 %x, i64* %x.addr, align 8
  store i64 %y, i64* %y.addr, align 8
  %0 = load i64, i64* %x.addr, align 8
  %shr = lshr i64 %0, 1
  %1 = load i64, i64* %x.addr, align 8
  %2 = load i64, i64* %y.addr, align 8
  %call = call i64 @parity(i64 %1, i64 %2)
  %shl = shl i64 %call, 63
  %or = or i64 %shr, %shl
  store i64 %or, i64* %x.addr, align 8
  %3 = load i64, i64* %x.addr, align 8
  ret i64 %3
}

declare dso_local i64 @parity(i64, i64)
