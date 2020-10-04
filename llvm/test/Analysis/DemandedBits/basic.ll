; RUN: opt -S -demanded-bits -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s
 
; CHECK-DAG: DemandedBits: 0xff for   %1 = add nsw i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for   %3 = trunc i32 %2 to i8
; CHECK-DAG: DemandedBits: 0xff for   %2 = mul nsw i32 %1, %b
define i8 @test_mul(i32 %a, i32 %b) {
  %1 = add nsw i32 %a, 5
  %2 = mul nsw i32 %1, %b
  %3 = trunc i32 %2 to i8
  ret i8 %3
}
