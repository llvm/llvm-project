; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

declare i32  @llvm.fshl.i32 (i32 %a, i32 %b, i32 %c)

declare i32  @llvm.fshr.i32 (i32 %a, i32 %b, i32 %c)


define i8 @test_fshr_const_amount_4(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshr_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for %fshr in   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0x0 for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0xff0 for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0x1f for 4 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 4)
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 4)
  %tr = trunc i32 %fshr to i8
  ret i8 %tr
}

define i8 @test_fshr_const_amount_5(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshr_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for %fshr in   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0x0 for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0x1fe0 for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0x1f for 5 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 5)
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 5)
  %tr = trunc i32 %fshr to i8
  ret i8 %tr


}

define i8 @test_fshr_const_amount_8(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshr_const_amount_8'
; CHECK-DAG: DemandedBits: 0xff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0x0 for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xff00 for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0x1f for 8 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for %fshr in   %tr = trunc i32 %fshr to i8
;
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
  %tr = trunc i32 %fshr to i8
  ret i8 %tr
}

define i8 @test_fshr_const_amount_9(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshr_const_amount_9'
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for %fshr in   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0x0 for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xff00 for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0x1f for 8 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
;
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 8)
  %tr = trunc i32 %fshr to i8
  ret i8 %tr
}

define i8 @test_fshr(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshr'
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for %fshr in   %tr = trunc i32 %fshr to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
;
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  %tr = trunc i32 %fshr to i8
  ret i8 %tr
}

define i8 @test_fshr_range_1(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_lshr_range_1'
; CHECK-DAG:  DemandedBits: 0xff for   %shl.t = trunc i32 %fshr to i8
; CHECK-DAG:  DemandedBits: 0xff for %fshr in   %shl.t = trunc i32 %fshr to i8
; CHECK-DAG:  DemandedBits: 0x1f for   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0x1f for 3 in   %b2 = and i32 %b, 3
; CHECK-DAG:  DemandedBits: 0xff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG:  DemandedBits: 0x7ff for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG:  DemandedBits: 0x1f for %b2 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG:  DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
;
  %b2 = and i32 %b, 3
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
  %shl.t = trunc i32 %fshr to i8
  ret i8 %shl.t
}

define i32 @test_lshr_range_2(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_lshr_range_2'
; CHECK-DAG: DemandedBits:  0x1f for   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0x1f for 3 in   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0xffffffff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0x1f for %b2 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
;
  %b2 = and i32 %b, 3
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %b2)
  ret i32 %fshr
}

define i32 @test_fshr_range_3(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshr_range_3'
; CHECK-DAG: DemandedBits: 0xffff for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for   %shl = shl i32 %fshr, 16
; CHECK-DAG: DemandedBits: 0xffff for %fshr in   %shl = shl i32 %fshr, 16
; CHECK-DAG: DemandedBits: 0xffffffff for 16 in   %shl = shl i32 %fshr, 16
;
  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  %shl = shl i32 %fshr, 16
  ret i32 %shl
}

define i32 @test_fshr_range_4(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshr_range_4'
; CHECK-DAG: DemandedBits: 0xffffff00 for   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for   %shr = ashr i32 %fshr, 8
; CHECK-DAG: DemandedBits: 0xffffff00 for %fshr in   %shr = ashr i32 %fshr, 8
; CHECK-DAG: DemandedBits: 0xffffffff for 8 in   %shr = ashr i32 %fshr, 8

  %fshr = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  %shr = ashr i32 %fshr, 8
  ret i32 %shr
}

define i32 @test_lshr_range_5(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_lshr_range_5'
; CHECK-DAG:  DemandedBits: 0xff for   %1 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG:  DemandedBits: 0xffffffff for %a in   %1 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG:  DemandedBits: 0xffffffff for %b in   %1 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG:  DemandedBits: 0x1f for %c in   %1 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG:  DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %1 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG:  DemandedBits: 0xffffffff for   %2 = and i32 %1, 255
; CHECK-DAG:  DemandedBits: 0xff for %1 in   %2 = and i32 %1, 255
; CHECK-DAG:  DemandedBits: 0xffffffff for 255 in   %2 = and i32 %1, 255
;
  %1 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  %2 = and i32 %1, 255
  ret i32 %2
}

define i32 @test_fshr_range_6(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshr_range_6'
; CHECK-DAG: DemandedBits: 0xffffffff for   %fshr.2 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0xffff for %a in   %fshr.2 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0xffff0000 for %b in   %fshr.2 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0x1f for 16 in   %fshr.2 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %fshr.2 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 16)
;
  %fshr.1 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  %fshr.2 = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 16)

  ret i32 %fshr.2
}

define i8 @test_lshr_var_amount(i32 %a, i32 %b, i32 %c){
; CHECK-LABEL: 'test_lshr_var_amount'
; CHECK-DAG: DemandedBits: 0x1f for   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0x1f for %a in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0x1f for %b in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0x1f for   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0x1f for %2 in   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0x1f for   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0x1f for %1 in   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for   %4 = call i32 @llvm.fshr.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %4 = call i32 @llvm.fshr.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0xffffffff for %c in   %4 = call i32 @llvm.fshr.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0x1f for %3 in   %4 = call i32 @llvm.fshr.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshr.i32 in   %4 = call i32 @llvm.fshr.i32(i32 %a, i32 %c, i32 %3)
;
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = call i32 @llvm.fshr.i32(i32 %a, i32 %c, i32 %3)
  %5 = trunc i32 %4 to i8
  ret i8 %5
}
