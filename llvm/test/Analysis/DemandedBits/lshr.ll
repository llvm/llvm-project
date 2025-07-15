; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

define i8 @test_lshr_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_lshr_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for %lshr in   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for   %lshr = lshr i32 %a, 4
; CHECK-DAG: DemandedBits: 0xff0 for %a in   %lshr = lshr i32 %a, 4
; CHECK-DAG: DemandedBits: 0xffffffff for 4 in   %lshr = lshr i32 %a, 4
;
  %lshr = lshr i32 %a, 4
  %lshr.t = trunc i32 %lshr to i8
  ret i8 %lshr.t
}

define i8 @test_lshr_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_lshr_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %lshr = lshr i32 %a, 5
; CHECK-DAG: DemandedBits: 0x1fe0 for %a in   %lshr = lshr i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for 5 in   %lshr = lshr i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for %lshr in   %lshr.t = trunc i32 %lshr to i8
;
  %lshr = lshr i32 %a, 5
  %lshr.t = trunc i32 %lshr to i8
  ret i8 %lshr.t
}
define i8 @test_lshr_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_lshr_const_amount_8'
; CHECK-DAG: DemandedBits: 0xff for   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for %lshr in   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for   %lshr = lshr i32 %a, 8
; CHECK-DAG: DemandedBits: 0xff00 for %a in   %lshr = lshr i32 %a, 8
; CHECK-DAG: DemandedBits: 0xffffffff for 8 in   %lshr = lshr i32 %a, 8
;
  %lshr = lshr i32 %a, 8
  %lshr.t = trunc i32 %lshr to i8
  ret i8 %lshr.t
}

define i8 @test_lshr_const_amount_9(i32 %a) {
; CHECK-LABEL: 'test_lshr_const_amount_9'
; CHECK-DAG: DemandedBits: 0xff for   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for %lshr in   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for   %lshr = lshr i32 %a, 9
; CHECK-DAG: DemandedBits: 0x1fe00 for %a in   %lshr = lshr i32 %a, 9
; CHECK-DAG: DemandedBits: 0xffffffff for 9 in   %lshr = lshr i32 %a, 9
;
  %lshr = lshr i32 %a, 9
  %lshr.t = trunc i32 %lshr to i8
  ret i8 %lshr.t
}

define i8 @test_lshr(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_lshr'
; CHECK-DAG: DemandedBits: 0xff for   %lshr = lshr i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %lshr = lshr i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %lshr = lshr i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for   %lshr.t = trunc i32 %lshr to i8
; CHECK-DAG: DemandedBits: 0xff for %lshr in   %lshr.t = trunc i32 %lshr to i8
;
  %lshr = lshr i32 %a, %b
  %lshr.t = trunc i32 %lshr to i8
  ret i8 %lshr.t
}
DemandedBits: 0xff for   %4 = lshr i32 %1, %3
DemandedBits: 0xffffffff for %1 in   %4 = lshr i32 %1, %3
DemandedBits: 0xffffffff for %3 in   %4 = lshr i32 %1, %3
DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
DemandedBits: 0xffffffff for   %3 = zext i8 %2 to i32
DemandedBits: 0xff for %2 in   %3 = zext i8 %2 to i32
DemandedBits: 0xffffffff for   %1 = add nsw i32 %a, %b
DemandedBits: 0xffffffff for %a in   %1 = add nsw i32 %a, %b
DemandedBits: 0xffffffff for %b in   %1 = add nsw i32 %a, %b
DemandedBits: 0xff for   %2 = trunc i32 %1 to i8
DemandedBits: 0xff for %1 in   %2 = trunc i32 %1 to i8

define i8 @test_lshr_var_amount(i32 %a, i32 %b){
; CHECK-LABEL: 'test_lshr_var_amount'
; CHECK-DAG: DemandedBits: 0xff for   %4 = lshr i32 %1, %3
; CHECK-DAG: DemandedBits: 0xffffffff for %1 in   %4 = lshr i32 %1, %3
; CHECK-DAG: DemandedBits: 0xffffffff for %3 in   %4 = lshr i32 %1, %3
; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xffffffff for   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0xff for %2 in   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0xffffffff for   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = trunc i32 %1 to i8
;
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = lshr i32 %1, %3
  %5 = trunc i32 %4 to i8
  ret i8 %5
}

define i8 @test_lshr_var_amount_exact(i32 %a, i32 %b){
 ; CHECK-LABEL 'test_lshr_var_amount_nsw'
 ; CHECK-DAG: DemandedBits: 0xffffffff for   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xff for   %2 = trunc i32 %1 to i8
 ; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = trunc i32 %1 to i8
 ; CHECK-DAG: DemandedBits: 0xffffffff for   %3 = zext i8 %2 to i32
 ; CHECK-DAG: DemandedBits: 0xff for %2 in   %3 = zext i8 %2 to i32
 ; CHECK-DAG: DemandedBits: 0xff for   %4 = lshr exact i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xffffffff for %1 in   %4 = lshr exact i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xffffffff for %3 in   %4 = lshr exact i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
 ; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
 ;
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = lshr exact i32 %1, %3
  %5 = trunc i32 %4 to i8
  ret i8 %5
}
