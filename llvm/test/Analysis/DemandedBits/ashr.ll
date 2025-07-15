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
