target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"

; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: call i32 @llvm.bswap.i32(i32 %i)
define i32 @test1(i32 %i) {
  %tmp1 = lshr i32 %i, 24
  %tmp3 = lshr i32 %i, 8
  %tmp4 = and i32 %tmp3, 65280
  %tmp5 = or i32 %tmp1, %tmp4
  %tmp7 = shl i32 %i, 8
  %tmp8 = and i32 %tmp7, 16711680
  %tmp9 = or i32 %tmp5, %tmp8
  %tmp11 = shl i32 %i, 24
  %tmp12 = or i32 %tmp9, %tmp11
  ret i32 %tmp12
}

; CHECK-LABEL: @test2
; CHECK: call i32 @llvm.bswap.i32(i32 %arg)
define i32 @test2(i32 %arg) {
  %tmp2 = shl i32 %arg, 24
  %tmp4 = shl i32 %arg, 8
  %tmp5 = and i32 %tmp4, 16711680
  %tmp6 = or i32 %tmp2, %tmp5
  %tmp8 = lshr i32 %arg, 8
  %tmp9 = and i32 %tmp8, 65280
  %tmp10 = or i32 %tmp6, %tmp9
  %tmp12 = lshr i32 %arg, 24
  %tmp14 = or i32 %tmp10, %tmp12
  ret i32 %tmp14
}

; CHECK-LABEL: @test3
; CHECK: call i16 @llvm.bswap.i16(i16 %s)
define i16 @test3(i16 %s) {
  %tmp2 = lshr i16 %s, 8
  %tmp4 = shl i16 %s, 8
  %tmp5 = or i16 %tmp2, %tmp4
  ret i16 %tmp5
}

; CHECK-LABEL: @test4
; CHECK: call i16 @llvm.bswap.i16(i16 %s)
define i16 @test4(i16 %s) {
  %tmp2 = lshr i16 %s, 8
  %tmp4 = shl i16 %s, 8
  %tmp5 = or i16 %tmp4, %tmp2
  ret i16 %tmp5
}

; CHECK-LABEL: @test5
; CHECK: call i16 @llvm.bswap.i16(i16 %a)
define i16 @test5(i16 %a) {
  %tmp = zext i16 %a to i32
  %tmp1 = and i32 %tmp, 65280
  %tmp2 = ashr i32 %tmp1, 8
  %tmp2.upgrd.1 = trunc i32 %tmp2 to i16
  %tmp4 = and i32 %tmp, 255
  %tmp5 = shl i32 %tmp4, 8
  %tmp5.upgrd.2 = trunc i32 %tmp5 to i16
  %tmp.upgrd.3 = or i16 %tmp2.upgrd.1, %tmp5.upgrd.2
  %tmp6 = bitcast i16 %tmp.upgrd.3 to i16
  %tmp6.upgrd.4 = zext i16 %tmp6 to i32
  %retval = trunc i32 %tmp6.upgrd.4 to i16
  ret i16 %retval
}

; PR2842
; CHECK-LABEL: @test6
; CHECK: call i32 @llvm.bswap.i32(i32 %x)
define i32 @test6(i32 %x) nounwind readnone {
  %tmp = shl i32 %x, 16
  %x.mask = and i32 %x, 65280
  %tmp1 = lshr i32 %x, 16
  %tmp2 = and i32 %tmp1, 255
  %tmp3 = or i32 %x.mask, %tmp
  %tmp4 = or i32 %tmp3, %tmp2
  %tmp5 = shl i32 %tmp4, 8
  %tmp6 = lshr i32 %x, 24
  %tmp7 = or i32 %tmp5, %tmp6
  ret i32 %tmp7
}

; PR23863
; CHECK-LABEL: @test7
; CHECK: call i32 @llvm.bswap.i32(i32 %x)
define i32 @test7(i32 %x) {
  %shl = shl i32 %x, 16
  %shr = lshr i32 %x, 16
  %or = or i32 %shl, %shr
  %and2 = shl i32 %or, 8
  %shl3 = and i32 %and2, -16711936
  %and4 = lshr i32 %or, 8
  %shr5 = and i32 %and4, 16711935
  %or6 = or i32 %shl3, %shr5
  ret i32 %or6
}

; CHECK-LABEL: @test8
; CHECK: call i16 @llvm.bswap.i16(i16 %a)
define i16 @test8(i16 %a) {
entry:
  %conv = zext i16 %a to i32
  %shr = lshr i16 %a, 8
  %shl = shl i32 %conv, 8
  %conv1 = zext i16 %shr to i32
  %or = or i32 %conv1, %shl
  %conv2 = trunc i32 %or to i16
  ret i16 %conv2
}

; CHECK-LABEL: @test9
; CHECK: call i16 @llvm.bswap.i16(i16 %a)
define i16 @test9(i16 %a) {
entry:
  %conv = zext i16 %a to i32
  %shr = lshr i32 %conv, 8
  %shl = shl i32 %conv, 8
  %or = or i32 %shr, %shl
  %conv2 = trunc i32 %or to i16
  ret i16 %conv2
}

; CHECK-LABEL: @test10
; CHECK: trunc i32 %a to i16
; CHECK: call i16 @llvm.bswap.i16(i16 %trunc)
define i16 @test10(i32 %a) {
  %shr1 = lshr i32 %a, 8
  %and1 = and i32 %shr1, 255
  %and2 = shl i32 %a, 8
  %shl1 = and i32 %and2, 65280
  %or = or i32 %and1, %shl1
  %conv = trunc i32 %or to i16
  ret i16 %conv
}
