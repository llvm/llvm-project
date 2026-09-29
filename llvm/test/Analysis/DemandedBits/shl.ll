; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

define i8 @test_shl_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_shl_const_amount_4'
; CHECK-DAG:  DemandedBits: 0xff for %shl = shl i32 %a, 4
; CHECK-DAG:  DemandedBits: 0xf for %a in %shl = shl i32 %a, 4
; CHECK-DAG:  DemandedBits: 0xffffffff for 4 in %shl = shl i32 %a, 4
; CHECK-DAG:  DemandedBits: 0xff for %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl in %shl.t = trunc i32 %shl to i8
;
  %shl = shl i32 %a, 4
  %shl.t = trunc i32 %shl to i8
  ret i8 %shl.t
}

define i8 @test_shl_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_shl_const_amount_5'
; CHECK-DAG:  DemandedBits: 0xff for %shl = shl i32 %a, 5
; CHECK-DAG:  DemandedBits: 0x7 for %a in %shl = shl i32 %a, 5
; CHECK-DAG:  DemandedBits: 0xffffffff for 5 in %shl = shl i32 %a, 5
; CHECK-DAG:  DemandedBits: 0xff for %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl in %shl.t = trunc i32 %shl to i8
;
  %shl = shl i32 %a, 5
  %shl.t = trunc i32 %shl to i8
  ret i8 %shl.t
}

define i8 @test_shl_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_shl_const_amount_8'
; CHECK-DAG:  DemandedBits: 0xff for %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl in %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl = shl i32 %a, 8
; CHECK-DAG:  DemandedBits: 0x0 for %a in %shl = shl i32 %a, 8
; CHECK-DAG:  DemandedBits: 0xffffffff for 8 in %shl = shl i32 %a, 8
;
  %shl = shl i32 %a, 8
  %shl.t = trunc i32 %shl to i8
  ret i8 %shl.t
}

define i8 @test_shl_const_amount_9(i32 %a) {
; CHECK-LABEL: 'test_shl_const_amount_9'
; CHECK-DAG:  DemandedBits: 0xff for %shl = shl i32 %a, 9
; CHECK-DAG:  DemandedBits: 0x0 for %a in %shl = shl i32 %a, 9
; CHECK-DAG:  DemandedBits: 0xffffffff for 9 in %shl = shl i32 %a, 9
; CHECK-DAG:  DemandedBits: 0xff for %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl in %shl.t = trunc i32 %shl to i8
;
  %shl = shl i32 %a, 9
  %shl.t = trunc i32 %shl to i8
  ret i8 %shl.t
}

define i8 @test_shl(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_shl'
; CHECK-DAG:  DemandedBits: 0xff for %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl in %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xff for %a in %shl = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in %shl = shl i32 %a, %b
;
  %shl = shl i32 %a, %b
  %shl.t = trunc i32 %shl to i8
  ret i8 %shl.t
}

define i8 @test_shl_range_1(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_shl_range_1'
; CHECK-DAG:  DemandedBits: 0xff for   %shl = shl i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xff for %a in   %shl = shl i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xffffffff for %b2 in   %shl = shl i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xff for   %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xff for %shl in   %shl.t = trunc i32 %shl to i8
; CHECK-DAG:  DemandedBits: 0xffffffff for   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xffffffff for 3 in   %b2 = and i32 %b, 3
;
  %b2 = and i32 %b, 3
  %shl = shl i32 %a, %b2
  %shl.t = trunc i32 %shl to i8
  ret i8 %shl.t
}

define i32 @test_shl_range_2(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_shl_range_2'
; CHECK-DAG:  DemandedBits: 0xffffffff for   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xffffffff for 3 in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xffffffff for   %shl = shl i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %shl = shl i32 %a, %b2
; CHECK-DAG:  DemandedBits: 0xffffffff for %b2 in   %shl = shl i32 %a, %b2
;
  %b2 = and i32 %b, 3
  %shl = shl i32 %a, %b2
  ret i32 %shl
}

define i32 @test_shl_range_3(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_shl_range_3'
; CHECK-DAG:  DemandedBits: 0xffffffff for   %shr = lshr i32 %shl, 16
; CHECK-DAG:  DemandedBits: 0xffff0000 for %shl in   %shr = lshr i32 %shl, 16
; CHECK-DAG:  DemandedBits: 0xffffffff for 16 in   %shr = lshr i32 %shl, 16
; CHECK-DAG:  DemandedBits: 0xffff0000 for   %shl = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %shl = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %shl = shl i32 %a, %b
;
  %shl = shl i32 %a, %b
  %shr = lshr i32 %shl, 16
  ret i32 %shr
}

define i32 @test_shl_range_4(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_shl_range_4'
; CHECK-DAG:  DemandedBits: 0xffffffff for   %shr = ashr i32 %shl, 8
; CHECK-DAG:  DemandedBits: 0xffffff00 for %shl in   %shr = ashr i32 %shl, 8
; CHECK-DAG:  DemandedBits: 0xffffffff for 8 in   %shr = ashr i32 %shl, 8
; CHECK-DAG:  DemandedBits: 0xffffff00 for   %shl = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %shl = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %shl = shl i32 %a, %b
  %shl = shl i32 %a, %b
  %shr = ashr i32 %shl, 8
  ret i32 %shr
}

define i32 @test_shl_range_5(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_shl_range_5'
; CHECK-DAG:  DemandedBits: 0xff for   %1 = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xff for %a in   %1 = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %1 = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for   %2 = and i32 %1, 255
; CHECK-DAG:  DemandedBits: 0xff for %1 in   %2 = and i32 %1, 255
; CHECK-DAG:  DemandedBits: 0xffffffff for 255 in   %2 = and i32 %1, 255
;
  %1 = shl i32 %a, %b
  %2 = and i32 %1, 255
  ret i32 %2
}

define i32 @test_shl_range_6(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_shl_range_6'
; CHECK-DAG:  DemandedBits: 0xffffffff for   %shl.2 = shl i32 %shl.1, 16
; CHECK-DAG:  DemandedBits: 0xffff for %shl.1 in   %shl.2 = shl i32 %shl.1, 16
; CHECK-DAG:  DemandedBits: 0xffffffff for 16 in   %shl.2 = shl i32 %shl.1, 16
; CHECK-DAG:  DemandedBits: 0xffff for   %shl.1 = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffff for %a in   %shl.1 = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %shl.1 = shl i32 %a, %b
;
  %shl.1 = shl i32 %a, %b
  %shl.2 = shl i32 %shl.1, 16
  ret i32 %shl.2
}

define i8 @test_shl_var_amount(i32 %a, i32 %b){
; CHECK-LABEL: 'test_shl_var_amount'
; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for   %4 = shl i32 %1, %3
; CHECK-DAG: DemandedBits: 0xff for %1 in   %4 = shl i32 %1, %3
; CHECK-DAG: DemandedBits: 0xffffffff for %3 in   %4 = shl i32 %1, %3
; CHECK-DAG: DemandedBits: 0xff for   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0xffffffff for   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0xff for %2 in   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0xff for   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for %a in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for %b in   %1 = add nsw i32 %a, %b
;
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = shl i32 %1, %3
  %5 = trunc i32 %4 to i8
  ret i8 %5
}

define i8 @test_shl_var_amount_nsw(i32 %a, i32 %b){
 ; CHECK-LABEL 'test_shl_var_amount_nsw'
 ; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
 ; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
 ; CHECK-DAG: DemandedBits: 0xff for   %4 = shl nsw i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xffffffff for %1 in   %4 = shl nsw i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xffffffff for %3 in   %4 = shl nsw i32 %1, %3
 ; CHECK-DAG: DemandedBits: 0xffffffff for   %3 = zext i8 %2 to i32
 ; CHECK-DAG: DemandedBits: 0xff for %2 in   %3 = zext i8 %2 to i32
 ; CHECK-DAG: DemandedBits: 0xff for   %2 = trunc i32 %1 to i8
 ; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = trunc i32 %1 to i8
 ; CHECK-DAG: DemandedBits: 0xffffffff for   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %1 = add nsw i32 %a, %b
 ; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %1 = add nsw i32 %a, %b
 ;
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = shl nsw i32 %1, %3
  %5 = trunc i32 %4 to i8
  ret i8 %5
}
