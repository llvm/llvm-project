; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

declare i32  @llvm.fshl.i32 (i32 %a, i32 %b, i32 %c)


define i8 @test_fshl_const_amount_4(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshl_const_amount_4'
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for %fshl in   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0xf for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0xf0000000 for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0x1f for 4 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 4)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 4)
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 4)
  %tr = trunc i32 %fshl to i8
  ret i8 %tr
}

define i8 @test_fshl_const_amount_5(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshl_const_amount_5'
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for %fshl in   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0x7 for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0xf8000000 for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0x1f for 5 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 5)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 5)
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 5)
  %tr = trunc i32 %fshl to i8
  ret i8 %tr


}

define i8 @test_fshl_const_amount_8(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshl_const_amount_8'
;
; CHECK-DAG: DemandedBits: 0xff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0x0 for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xff000000 for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0x1f for 8 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for %fshl in   %tr = trunc i32 %fshl to i8
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)
  %tr = trunc i32 %fshl to i8
  ret i8 %tr
}

define i8 @test_fshl_const_amount_9(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshl_const_amount_9'
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for %fshl in   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 9)
; CHECK-DAG: DemandedBits: 0x0 for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 9)
; CHECK-DAG: DemandedBits: 0x7f800000 for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 9)
; CHECK-DAG: DemandedBits: 0x1f for 9 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 9)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 9)

  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 9)
  %tr = trunc i32 %fshl to i8
  ret i8 %tr
}

define i8 @test_fshl(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshl'
;
; CHECK-DAG: DemandedBits: 0xff for   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for %fshl in   %tr = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %tr = trunc i32 %fshl to i8
  ret i8 %tr
}

define i8 @test_fshl_range_1(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshl_range_1'
; CHECK-DAG: DemandedBits: 0xff for   %shl.t = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0xff for %fshl in   %shl.t = trunc i32 %fshl to i8
; CHECK-DAG: DemandedBits: 0x1f for   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0x1f for 3 in   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0xff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xff for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0x1f for %b2 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
;
  %b2 = and i32 %b, 3
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
  %shl.t = trunc i32 %fshl to i8
  ret i8 %shl.t
}

define i32 @test_fshl_range_2(i32 %a, i32 %b) {
; CHECK-LABEL: 'test_fshl_range_2'
; CHECK-DAG: DemandedBits: 0xffffffff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0x1f for %b2 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
; CHECK-DAG: DemandedBits: 0x1f for   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0x3 for %b in   %b2 = and i32 %b, 3
; CHECK-DAG: DemandedBits: 0x1f for 3 in   %b2 = and i32 %b, 3
;
  %b2 = and i32 %b, 3
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %b2)
  ret i32 %fshl
}

define i32 @test_fshl_range_3(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshl_range_3'
; CHECK-DAG: DemandedBits: 0xffff for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for   %shl = shl i32 %fshl, 16
; CHECK-DAG: DemandedBits: 0xffff for %fshl in   %shl = shl i32 %fshl, 16
; CHECK-DAG: DemandedBits: 0xffffffff for 16 in   %shl = shl i32 %fshl, 16
;
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %shl = shl i32 %fshl, 16
  ret i32 %shl
}

define i32 @test_fshl_range_4(i32 %a, i32 %b, i32 %c) {
; CHECK-DAG: DemandedBits: 0xffffff00 for   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for   %shr = ashr i32 %fshl, 8
; CHECK-DAG: DemandedBits: 0xffffff00 for %fshl in   %shr = ashr i32 %fshl, 8
; CHECK-DAG: DemandedBits: 0xffffffff for 8 in   %shr = ashr i32 %fshl, 8


; CHECK-LABEL: 'test_fshl_range_4'
  %fshl = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %shr = ashr i32 %fshl, 8
  ret i32 %shr
}

define i32 @test_fshl_range_5(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshl_range_5'
; CHECK-DAG: DemandedBits: 0xff for   %1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for   %2 = and i32 %1, 255
; CHECK-DAG: DemandedBits: 0xff for %1 in   %2 = and i32 %1, 255
; CHECK-DAG: DemandedBits: 0xffffffff for 255 in   %2 = and i32 %1, 255
;
  %1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %2 = and i32 %1, 255
  ret i32 %2
}

define i32 @test_fshl_range_6(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: 'test_fshl_range_6'
; CHECK-DAG: DemandedBits: 0xffff for   %fshl.1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %fshl.1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for %b in   %fshl.1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0x1f for %c in   %fshl.1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl.1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; CHECK-DAG: DemandedBits: 0xffffffff for   %fshl.2 = call i32 @llvm.fshl.i32(i32 %fshl.1, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0xffff for %fshl.1 in   %fshl.2 = call i32 @llvm.fshl.i32(i32 %fshl.1, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0xffff0000 for %b in   %fshl.2 = call i32 @llvm.fshl.i32(i32 %fshl.1, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0x1f for 16 in   %fshl.2 = call i32 @llvm.fshl.i32(i32 %fshl.1, i32 %b, i32 16)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %fshl.2 = call i32 @llvm.fshl.i32(i32 %fshl.1, i32 %b, i32 16)
;
  %fshl.1 = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  %fshl.2 = call i32 @llvm.fshl.i32(i32 %fshl.1, i32 %b, i32 16)

  ret i32 %fshl.2
}

define i8 @test_fshl_var_amount(i32 %a, i32 %b, i32 %c){
; CHECK-LABEL: 'test_fshl_var_amount'
; CHECK-DAG: DemandedBits: 0xff for   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for %4 in   %5 = trunc i32 %4 to i8
; CHECK-DAG: DemandedBits: 0xff for   %4 = call i32 @llvm.fshl.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0xffffffff for %a in   %4 = call i32 @llvm.fshl.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0xffffffff for %c in   %4 = call i32 @llvm.fshl.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0x1f for %3 in   %4 = call i32 @llvm.fshl.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0xffffffffffffffff for @llvm.fshl.i32 in   %4 = call i32 @llvm.fshl.i32(i32 %a, i32 %c, i32 %3)
; CHECK-DAG: DemandedBits: 0x1f for   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0x1f for %a in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0x1f for %b in   %1 = add nsw i32 %a, %b
; CHECK-DAG: DemandedBits: 0x1f for   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0x1f for %2 in   %3 = zext i8 %2 to i32
; CHECK-DAG: DemandedBits: 0x1f for   %2 = trunc i32 %1 to i8
; CHECK-DAG: DemandedBits: 0x1f for %1 in   %2 = trunc i32 %1 to i8
  %1 = add nsw i32 %a, %b
  %2 = trunc i32 %1 to i8
  %3 = zext i8 %2 to i32
  %4 = call i32 @llvm.fshl.i32(i32 %a, i32 %c, i32 %3)
  %5 = trunc i32 %4 to i8
  ret i8 %5
}
