; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

define i8 @test_ashr_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_ashr_const_amount_4'
; CHECK-DAG:  DemandedBits: 0xff for   %ashr = ashr i32 %a, 4
; CHECK-DAG:  DemandedBits: 0xff0 for %a in   %ashr = ashr i32 %a, 4
; CHECK-DAG:  DemandedBits: 0xffffffff for 4 in   %ashr = ashr i32 %a, 4
; CHECK-DAG:  DemandedBits: 0xff for   %ashr.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xff for %ashr in   %ashr.t = trunc i32 %ashr to i8
;
  %ashr = ashr i32 %a, 4
  %ashr.t = trunc i32 %ashr to i8
  ret i8 %ashr.t
}

define i8 @test_ashr_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_ashr_const_amount_5'
; CHECK-DAG:  DemandedBits: 0xff for   %ashr = ashr i32 %a, 5
; CHECK-DAG:  DemandedBits: 0x1fe0 for %a in   %ashr = ashr i32 %a, 5
; CHECK-DAG:  DemandedBits: 0xffffffff for 5 in   %ashr = ashr i32 %a, 5
; CHECK-DAG:  DemandedBits: 0xff for   %ashr.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xff for %ashr in   %ashr.t = trunc i32 %ashr to i8
;
  %ashr = ashr i32 %a, 5
  %ashr.t = trunc i32 %ashr to i8
  ret i8 %ashr.t
}

define i8 @test_ashr_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_ashr_const_amount_8'
; CHECK-DAG:  DemandedBits: 0xff for   %ashr = ashr i32 %a, 8
; CHECK-DAG:  DemandedBits: 0xff00 for %a in   %ashr = ashr i32 %a, 8
; CHECK-DAG:  DemandedBits: 0xffffffff for 8 in   %ashr = ashr i32 %a, 8
; CHECK-DAG:  DemandedBits: 0xff for   %ashr.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xff for %ashr in   %ashr.t = trunc i32 %ashr to i8
;
  %ashr = ashr i32 %a, 8
  %ashr.t = trunc i32 %ashr to i8
  ret i8 %ashr.t
}

define i8 @test_ashr_const_amount_9(i32 %a) {

; CHECK-LABEL: 'test_ashr_const_amount_9'
; CHECK-DAG:  DemandedBits: 0xff for   %ashr.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xff for %ashr in   %ashr.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xff for   %ashr = ashr i32 %a, 8
; CHECK-DAG:  DemandedBits: 0xff00 for %a in   %ashr = ashr i32 %a, 8
; CHECK-DAG:  DemandedBits: 0xffffffff for 8 in   %ashr = ashr i32 %a, 8
;
  %ashr = ashr i32 %a, 8
  %ashr.t = trunc i32 %ashr to i8
  ret i8 %ashr.t
}

define i8 @test_ashr(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_ashr'
; CHECK-DAG:  DemandedBits: 0xff for   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xff for   %ashr.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xff for %ashr in   %ashr.t = trunc i32 %ashr to i8
;
  %ashr = ashr i32 %a, %b
  %ashr.t = trunc i32 %ashr to i8
  ret i8 %ashr.t
}

define i8 @test_ashr_range_1(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_ashr_range_1'
; CHECK-DAG:  DemandedBits: 0xff for   %shl.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xff for %ashr in   %shl.t = trunc i32 %ashr to i8
; CHECK-DAG:  DemandedBits: 0xffffffff for   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xffffffff for 3 in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xff for   %ashr = ashr i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0x7ff for %a in   %ashr = ashr i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xffffffff for %b2 in   %ashr = ashr i32 %a, %b2
;
  %b2 = and i32 %b, 3
  %ashr = ashr i32 %a, %b2
  %shl.t = trunc i32 %ashr to i8
  ret i8 %shl.t
}

define i32 @test_ashr_range_2(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_ashr_range_2'
; CHECK-DAG:  DemandedBits: 0xffffffff for   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xffffffff for 3 in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xffffffff for   %ashr = ashr i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %ashr = ashr i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xffffffff for %b2 in   %ashr = ashr i32 %a, %b2
;
  %b2 = and i32 %b, 3
  %ashr = ashr i32 %a, %b2
  ret i32 %ashr
}

define i32 @test_ashr_range_3(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_ashr_range_3'
; CHECK-DAG:  DemandedBits:  0xffff for   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits:  0xffffffff for %a in   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits:  0xffffffff for %b in   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits:  0xffffffff for   %shl = shl i32 %ashr, 16
; CHECK-DAG:  DemandedBits:  0xffff for %ashr in   %shl = shl i32 %ashr, 16
; CHECK-DAG:  DemandedBits:  0xffffffff for 16 in   %shl = shl i32 %ashr, 16
;
  %ashr = ashr i32 %a, %b
  %shl = shl i32 %ashr, 16
  ret i32 %shl
}
define i32 @test_ashr_range_4(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_ashr_range_4'
; CHECK-DAG:  DemandedBits: 0xffffffff for   %shr = lshr i32 %ashr, 8
; CHECK-DAG:  DemandedBits: 0xffffff00 for %ashr in   %shr = lshr i32 %ashr, 8
; CHECK-DAG:  DemandedBits: 0xffffffff for 8 in   %shr = lshr i32 %ashr, 8
; CHECK-DAG:  DemandedBits: 0xffffff00 for   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffff00 for %a in   %ashr = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %ashr = ashr i32 %a, %b
  %ashr = ashr i32 %a, %b
  %shr = lshr i32 %ashr, 8
  ret i32 %shr
}

define i32 @test_ashr_range_5(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_ashr_range_5'
; CHECK-DAG:  DemandedBits: 0xffffffff for   %2 = and i32 %1, 255
; CHECK-DAG:  DemandedBits: 0xff for %1 in   %2 = and i32 %1, 255
; CHECK-DAG:  DemandedBits: 0xffffffff for 255 in   %2 = and i32 %1, 255
; CHECK-DAG:  DemandedBits: 0xff for   %1 = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %1 = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %1 = ashr i32 %a, %b
;
  %1 = ashr i32 %a, %b
  %2 = and i32 %1, 255
  ret i32 %2
}

define i32 @test_ashr_range_6(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_ashr_range_6'
; CHECK-DAG:  DemandedBits: 0xffff0000 for   %lshr.1 = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffff0000 for %a in   %lshr.1 = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %lshr.1 = ashr i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for   %lshr.2 = ashr i32 %lshr.1, 16
; CHECK-DAG:  DemandedBits: 0xffff0000 for %lshr.1 in   %lshr.2 = ashr i32 %lshr.1, 16
; CHECK-DAG:  DemandedBits: 0xffffffff for 16 in   %lshr.2 = ashr i32 %lshr.1, 16
;
  %lshr.1 = ashr i32 %a, %b
  %lshr.2 = ashr i32 %lshr.1, 16
  ret i32 %lshr.2
}

define i8 @test_ashr_var_amount(i32 %a, i32 %b){
; CHECK-LABEL: 'test_ashr_var_amount'
; CHECK-DAG: DemandedBits: 0xff for   %4 = ashr i32 %1, %3
; CHECK-DAG: DemandedBits: 0xffffffff for %1 in   %4 = ashr i32 %1, %3
; CHECK-DAG: DemandedBits: 0xffffffff for %3 in   %4 = ashr i32 %1, %3
; CHECK-DAG: DemandedBits: 0xff for   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0xffffffff for   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0xff for %2 in   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
;
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = ashr i32 %1, %3
  %5 = trunc i32 %4 to i8
  ret i8 %5
}

define i8 @test_ashr_var_amount_nsw(i32 %a, i32 %b){
 ; CHECK-LABEL 'test_ashr_var_amount_nsw'
 ; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
 ; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
 ; CHECK-DAG: DemandedBits: 0xffffffff for   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xff for   %2 = trunc i32 %1 to i8
 ; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = trunc i32 %1 to i8
 ; CHECK-DAG: DemandedBits: 0xffffffff for   %3 = zext i8 %2 to i32
 ; CHECK-DAG: DemandedBits: 0xff for %2 in   %3 = zext i8 %2 to i32
 ; CHECK-DAG: DemandedBits: 0xff for   %4 = ashr exact i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xffffffff for %1 in   %4 = ashr exact i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xffffffff for %3 in   %4 = ashr exact i32 %1, %3
 ;
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = ashr exact i32 %1, %3
  %5 = trunc i32 %4 to i8
  ret i8 %5
}
