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
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in %shl = shl i32 %a, %b
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in %shl = shl i32 %a, %b
;
  %shl = shl i32 %a, %b
  %shl.t = trunc i32 %shl to i8
  ret i8 %shl.t
}
