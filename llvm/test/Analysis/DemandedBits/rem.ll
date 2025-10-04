; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s


define i8 @test_srem_zext_trunc_const_amount2(i8 %a) {
; CHECK-LABEL: 'test_srem_zext_trunc_const_amount2'
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %ext, 2
; CHECK-DAG: DemandedBits: 0x80000001 for %ext in   %rem = srem i32 %ext, 2
; CHECK-DAG: DemandedBits: 0x80000001 for 2 in   %rem = srem i32 %ext, 2
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0x80000001 for   %ext = sext i8 %a to i32
; CHECK-DAG: DemandedBits: 0x81 for %a in   %ext = sext i8 %a to i32
;
  %ext = sext i8 %a to i32
  %rem = srem i32 %ext, 2
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_srem_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_srem_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, 4
; CHECK-DAG: DemandedBits: 0x80000003 for %a in   %rem = srem i32 %a, 4
; CHECK-DAG: DemandedBits: 0x80000003 for 4 in   %rem = srem i32 %a, 4
;
  %rem = srem i32 %a, 4
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_srem_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_srem_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = srem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for 5 in   %rem = srem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
;
  %rem = srem i32 %a, 5
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_srem_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_srem_const_amount_8'
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, 8
; CHECK-DAG: DemandedBits: 0x80000007 for %a in   %rem = srem i32 %a, 8
; CHECK-DAG: DemandedBits: 0x80000007 for 8 in   %rem = srem i32 %a, 8
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
;
  %rem = srem i32 %a, 8
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_srem_const_amount_9(i32 %a) {
; CHECK-LABEL: 'test_srem_const_amount_9'
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, 9
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = srem i32 %a, 9
; CHECK-DAG: DemandedBits: 0xffffffff for 9 in   %rem = srem i32 %a, 9
;
  %rem = srem i32 %a, 9
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_srem(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_srem'
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = srem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %rem = srem i32 %a, %b
;
  %rem = srem i32 %a, %b
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}
