; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

define i8 @test_sdiv_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_sdiv_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for   %div = sdiv i32 %a, 4
; CHECK-DAG: DemandedBits: 0x3fc for %a in   %div = sdiv i32 %a, 4
; CHECK-DAG: DemandedBits: 0xffffffff for 4 in   %div = sdiv i32 %a, 4
;
  %div = sdiv i32 %a, 4
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_sdiv_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_sdiv_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %div = sdiv i32 %a, 5
; CHECK-DAG: DemandedBits: 0xfff for %a in   %div = sdiv i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for 5 in   %div = sdiv i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
;
  %div = sdiv i32 %a, 5
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_sdiv_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_sdiv_const_amount_8'
; CHECK-DAG: DemandedBits: 0xff for   %div = sdiv i32 %a, 8
; CHECK-DAG: DemandedBits: 0x7f8 for %a in   %div = sdiv i32 %a, 8
; CHECK-DAG: DemandedBits: 0xffffffff for 8 in   %div = sdiv i32 %a, 8
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
;
  %div = sdiv i32 %a, 8
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_sdiv_const_amount_9(i32 %a) {
; CHECK-LABEL: 'test_sdiv_const_amount_9'
; CHECK-DAG: DemandedBits: 0xff for   %div = udiv i32 %a, 9
; CHECK-DAG: DemandedBits: 0xfff for %a in   %div = udiv i32 %a, 9
; CHECK-DAG: DemandedBits: 0xffffffff for 9 in   %div = udiv i32 %a, 9
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
;
  %div = udiv i32 %a, 9
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_sdiv(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_sdiv'
; CHECK-DAG: DemandedBits: 0xff for   %div = sdiv i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %div = sdiv i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %div = sdiv i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
;
  %div = sdiv i32 %a, %b
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_udiv_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_udiv_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %div = udiv i32 %a, 4
; CHECK-DAG: DemandedBits: 0x3fc for %a in   %div = udiv i32 %a, 4
; CHECK-DAG: DemandedBits: 0xffffffff for 4 in   %div = udiv i32 %a, 4
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
;
  %div = udiv i32 %a, 4
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_udiv_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_udiv_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for   %div = udiv i32 %a, 5
; CHECK-DAG: DemandedBits: 0x7ff for %a in   %div = udiv i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for 5 in   %div = udiv i32 %a, 5
;
  %div = udiv i32 %a, 5
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_udiv_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_udiv_const_amount_8'
; CHECK-DAG: DemandedBits: 0xff for   %div = udiv i32 %a, 8
; CHECK-DAG: DemandedBits: 0x7f8 for %a in   %div = udiv i32 %a, 8
; CHECK-DAG: DemandedBits: 0xffffffff for 8 in   %div = udiv i32 %a, 8
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
;
  %div = udiv i32 %a, 8
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_udiv_const_amount_9(i32 %a) {
; CHECK-LABEL: 'test_udiv_const_amount_9'
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for   %div = udiv i32 %a, 9
; CHECK-DAG: DemandedBits: 0xfff for %a in   %div = udiv i32 %a, 9
; CHECK-DAG: DemandedBits: 0xffffffff for 9 in   %div = udiv i32 %a, 9
;
  %div = udiv i32 %a, 9
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_udiv(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_udiv'
; CHECK-DAG: DemandedBits: 0xff for   %div = udiv i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %div = udiv i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %div = udiv i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for   %div.t = trunc i32 %div to i8
; CHECK-DAG: DemandedBits: 0xff for %div in   %div.t = trunc i32 %div to i8
;
  %div = udiv i32 %a, %b
  %div.t = trunc i32 %div to i8
  ret i8 %div.t
}

define i8 @test_srem_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_srem_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, 4
; CHECK-DAG: DemandedBits: 0x3 for %a in   %rem = srem i32 %a, 4
; CHECK-DAG: DemandedBits: 0xffffffff for 4 in   %rem = srem i32 %a, 4
;
  %rem = srem i32 %a, 4
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_srem_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_srem_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = srem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for 5 in   %rem = srem i32 %a, 5
;
  %rem = srem i32 %a, 5
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_srem_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_srem_const_amount_8'
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, 8
; CHECK-DAG: DemandedBits: 0x7 for %a in   %rem = srem i32 %a, 8
; CHECK-DAG: DemandedBits: 0xffffffff for 8 in   %rem = srem i32 %a, 8
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
; CHECK-DAG: DemandedBits: 0xff for   %rem = srem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = srem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %rem = srem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
;
  %rem = srem i32 %a, %b
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_urem_const_amount_4(i32 %a) {
; CHECK-LABEL: 'test_urem_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem = urem i32 %a, 4
; CHECK-DAG: DemandedBits: 0x3 for %a in   %rem = urem i32 %a, 4
; CHECK-DAG: DemandedBits: 0xffffffff for 4 in   %rem = urem i32 %a, 4
;
  %rem = urem i32 %a, 4
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}






define i8 @test_urem_const_amount_5(i32 %a) {
; CHECK-LABEL: 'test_urem_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %rem = urem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = urem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xffffffff for 5 in   %rem = urem i32 %a, 5
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
;
  %rem = urem i32 %a, 5
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_urem_const_amount_8(i32 %a) {
; CHECK-LABEL: 'test_urem_const_amount_8'
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for   %rem = urem i32 %a, 8
; CHECK-DAG: DemandedBits: 0x7 for %a in   %rem = urem i32 %a, 8
; CHECK-DAG: DemandedBits: 0xffffffff for 8 in   %rem = urem i32 %a, 8
;
  %rem = urem i32 %a, 8
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_urem_const_amount_9(i32 %a) {
; CHECK-LABEL: 'test_urem_const_amount_9'
; CHECK-DAG: DemandedBits: 0xff for   %rem = urem i32 %a, 9
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = urem i32 %a, 9
; CHECK-DAG: DemandedBits: 0xffffffff for 9 in   %rem = urem i32 %a, 9
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
;
  %rem = urem i32 %a, 9
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}

define i8 @test_urem(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_urem'
; CHECK-DAG: DemandedBits: 0xff for   %rem = urem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %rem = urem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %rem = urem i32 %a, %b
; CHECK-DAG: DemandedBits: 0xff for   %rem.t = trunc i32 %rem to i8
; CHECK-DAG: DemandedBits: 0xff for %rem in   %rem.t = trunc i32 %rem to i8
;
  %rem = urem i32 %a, %b
  %rem.t = trunc i32 %rem to i8
  ret i8 %rem.t
}
