; RUN: opt -S -demanded-bits -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s
 
; CHECK-DAG: DemandedBits: 0xff for   %1 = add nsw i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for   %3 = trunc i32 %2 to i8
; CHECK-DAG: DemandedBits: 0xff for   %2 = mul nsw i32 %1, %b
; CHECK-DAG: DemandedBits: 0x1 for   %4 = trunc i32 %2 to i1
; CHECK-DAG: DemandedBits: 0xff for   %5 = zext i1 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for   %6 = add nsw i8 %3, %5
; CHECK-DAG: DemandedBits: 0xff for %a in   %1 = add nsw i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for 5 in   %1 = add nsw i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = mul nsw i32 %1, %b
; CHECK-DAG: DemandedBits: 0xff for %b in   %2 = mul nsw i32 %1, %b
; CHECK-DAG: DemandedBits: 0xff for %2 in   %3 = trunc i32 %2 to i8
; CHECK-DAG: DemandedBits: 0x1 for %2 in   %4 = trunc i32 %2 to i1
; CHECK-DAG: DemandedBits: 0x1 for %4 in   %5 = zext i1 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for %3 in   %6 = add nsw i8 %3, %5
; CHECK-DAG: DemandedBits: 0xff for %5 in   %6 = add nsw i8 %3, %5
define i8 @test_mul(i32 %a, i32 %b) {
  %1 = add nsw i32 %a, 5
  %2 = mul nsw i32 %1, %b
  %3 = trunc i32 %2 to i8
  %4 = trunc i32 %2 to i1
  %5 = zext i1 %4 to i8
  %6 = add nsw i8 %3, %5
  ret i8 %6
}
