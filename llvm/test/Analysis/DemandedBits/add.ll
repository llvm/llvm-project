; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

; CHECK-LABEL: Printing analysis 'Demanded Bits Analysis' for function 'test_add':
; CHECK-DAG: DemandedBits: 0x1e for   %1 = and i32 %a, 9
; CHECK-DAG: DemandedBits: 0x1a for   %2 = and i32 %b, 9
; CHECK-DAG: DemandedBits: 0x1a for   %3 = and i32 %c, 13
; CHECK-DAG: DemandedBits: 0x1a for   %4 = and i32 %d, 4
; CHECK-DAG: DemandedBits: 0x1a for   %5 = or i32 %2, %3
; CHECK-DAG: DemandedBits: 0x1a for   %6 = or i32 %4, %5
; CHECK-DAG: DemandedBits: 0x10 for   %7 = add i32 %1, %6
; CHECK-DAG: DemandedBits: 0xffffffff for   %8 = and i32 %7, 16
define i32 @test_add(i32 %a, i32 %b, i32 %c, i32 %d) {
  %1 = and i32 %a, 9
  %2 = and i32 %b, 9
  %3 = and i32 %c, 13
  %4 = and i32 %d, 4 ; no bit of %d alive, %4 simplifies to zero
  %5 = or i32 %2, %3
  %6 = or i32 %4, %5
  %7 = add i32 %1, %6
  %8 = and i32 %7, 16
  ret i32 %8
}
