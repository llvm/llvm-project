; RUN: opt -S -demanded-bits -analyze < %s | FileCheck %s

; CHECK-LABEL: 'test_mul'
; CHECK-DAG: DemandedBits: 0xFF for   %1 = add nsw i32 %a, 5
; CHECK-DAG: DemandedBits: 0xFF for   %3 = trunc i32 %2 to i8
; CHECK-DAG: DemandedBits: 0xFF for   %2 = mul nsw i32 %1, %b
define i8 @test_mul(i32 %a, i32 %b) {
  %1 = add nsw i32 %a, 5
  %2 = mul nsw i32 %1, %b
  %3 = trunc i32 %2 to i8
  ret i8 %3
}

; CHECK-LABEL: 'test_icmp1'
; CHECK-DAG: DemandedBits: 0x1 for   %3 = icmp eq i32 %1, %2
; CHECK-DAG: DemandedBits: 0xFFF for   %1 = and i32 %a, 255
; CHECK-DAG: DemandedBits: 0xFFF for   %2 = shl i32 %1, 4
define i1 @test_icmp1(i32 %a, i32 %b) {
  %1 = and i32 %a, 255
  %2 = shl i32 %1, 4
  %3 = icmp eq i32 %1, %2
  ret i1 %3
}

; CHECK-LABEL: 'test_icmp2'
; CHECK-DAG: DemandedBits: 0x1 for   %3 = icmp eq i32 %1, %2
; CHECK-DAG: DemandedBits: 0xFFF for   %1 = and i32 %a, 255
; CHECK-DAG: DemandedBits: 0xFF for   %2 = ashr i32 %1, 4
define i1 @test_icmp2(i32 %a, i32 %b) {
  %1 = and i32 %a, 255
  %2 = ashr i32 %1, 4
  %3 = icmp eq i32 %1, %2
  ret i1 %3
}

; CHECK-LABEL: 'test_icmp3'
; CHECK-DAG: DemandedBits: 0xFFFFFFFF for   %1 = and i32 %a, 255
; CHECK-DAG: DemandedBits: 0x1 for   %2 = icmp eq i32 -1, %1
define i1 @test_icmp3(i32 %a) {
  %1 = and i32 %a, 255
  %2 = icmp eq i32 -1, %1
  ret i1 %2
}
