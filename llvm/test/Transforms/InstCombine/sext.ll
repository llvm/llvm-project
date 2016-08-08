; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

declare i32 @llvm.ctpop.i32(i32)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.cttz.i32(i32, i1)

define i64 @test1(i32 %x) {
  %t = call i32 @llvm.ctpop.i32(i32 %x)
  %s = sext i32 %t to i64
  ret i64 %s
  
; CHECK-LABEL: @test1(
; CHECK: zext i32 %t
}

define i64 @test2(i32 %x) {
  %t = call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %s = sext i32 %t to i64
  ret i64 %s

; CHECK-LABEL: @test2(
; CHECK: zext i32 %t
}

define i64 @test3(i32 %x) {
  %t = call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %s = sext i32 %t to i64
  ret i64 %s

; CHECK-LABEL: @test3(
; CHECK: zext i32 %t
}

define i64 @test4(i32 %x) {
  %t = udiv i32 %x, 3
  %s = sext i32 %t to i64
  ret i64 %s

; CHECK-LABEL: @test4(
; CHECK: zext i32 %t
}

define i64 @test5(i32 %x) {
  %t = urem i32 %x, 30000
  %s = sext i32 %t to i64
  ret i64 %s
; CHECK-LABEL: @test5(
; CHECK: zext i32 %t
}

define i64 @test6(i32 %x) {
  %u = lshr i32 %x, 3
  %t = mul i32 %u, 3
  %s = sext i32 %t to i64
  ret i64 %s
; CHECK-LABEL: @test6(
; CHECK: zext i32 %t
}

define i64 @test7(i32 %x) {
  %t = and i32 %x, 511
  %u = sub i32 20000, %t
  %s = sext i32 %u to i64
  ret i64 %s
; CHECK-LABEL: @test7(
; CHECK: zext i32 %u to i64
}

define i32 @test8(i8 %a, i32 %f, i1 %p, i32* %z) {
  %d = lshr i32 %f, 24
  %e = select i1 %p, i32 %d, i32 0
  %s = trunc i32 %e to i16
  %n = sext i16 %s to i32
  ret i32 %n
; CHECK-LABEL: @test8(
; CHECK: %d = lshr i32 %f, 24
; CHECK: %n = select i1 %p, i32 %d, i32 0
; CHECK: ret i32 %n
}

; rdar://6013816
define i16 @test9(i16 %t, i1 %cond) nounwind {
entry:
	br i1 %cond, label %T, label %F
T:
	%t2 = sext i16 %t to i32
	br label %F

F:
	%V = phi i32 [%t2, %T], [42, %entry]
	%W = trunc i32 %V to i16
	ret i16 %W
; CHECK-LABEL: @test9(
; CHECK: T:
; CHECK-NEXT: br label %F
; CHECK: F:
; CHECK-NEXT: phi i16
; CHECK-NEXT: ret i16
}

; PR2638
define i32 @test10(i32 %i) nounwind  {
entry:
        %tmp12 = trunc i32 %i to i8
        %tmp16 = shl i8 %tmp12, 6
        %a = ashr i8 %tmp16, 6 
        %b = sext i8 %a to i32 
        ret i32 %b
; CHECK-LABEL: @test10(
; CHECK:  shl i32 %i, 30
; CHECK-NEXT: ashr exact i32
; CHECK-NEXT: ret i32
}

define void @test11(<2 x i16> %srcA, <2 x i16> %srcB, <2 x i16>* %dst) {
  %cmp = icmp eq <2 x i16> %srcB, %srcA
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %tmask = ashr <2 x i16> %sext, <i16 15, i16 15> 
  store <2 x i16> %tmask, <2 x i16>* %dst
  ret void                                                                                                                      
; CHECK-LABEL: @test11(
; CHECK-NEXT: icmp eq
; CHECK-NEXT: sext <2 x i1>
; CHECK-NEXT: store <2 x i16>
; CHECK-NEXT: ret
}                                                                                                                               

define i64 @test12(i32 %x) nounwind {
  %shr = lshr i32 %x, 1
  %sub = sub nsw i32 0, %shr
  %conv = sext i32 %sub to i64
  ret i64 %conv
; CHECK-LABEL: @test12(
; CHECK: sext
; CHECK: ret
}

define i32 @test13(i32 %x) nounwind {
  %and = and i32 %x, 8
  %cmp = icmp eq i32 %and, 0
  %ext = sext i1 %cmp to i32
  ret i32 %ext
; CHECK-LABEL: @test13(
; CHECK-NEXT: %and = lshr i32 %x, 3
; CHECK-NEXT: %1 = and i32 %and, 1
; CHECK-NEXT: %sext = add nsw i32 %1, -1
; CHECK-NEXT: ret i32 %sext
}

define i32 @test14(i16 %x) nounwind {
  %and = and i16 %x, 16
  %cmp = icmp ne i16 %and, 16
  %ext = sext i1 %cmp to i32
  ret i32 %ext
; CHECK-LABEL: @test14(
; CHECK-NEXT: %and = lshr i16 %x, 4
; CHECK-NEXT: %1 = and i16 %and, 1
; CHECK-NEXT: %sext = add nsw i16 %1, -1
; CHECK-NEXT: %ext = sext i16 %sext to i32
; CHECK-NEXT: ret i32 %ext
}

define i32 @test15(i32 %x) nounwind {
  %and = and i32 %x, 16
  %cmp = icmp ne i32 %and, 0
  %ext = sext i1 %cmp to i32
  ret i32 %ext
; CHECK-LABEL: @test15(
; CHECK-NEXT: %1 = shl i32 %x, 27
; CHECK-NEXT: %sext = ashr i32 %1, 31
; CHECK-NEXT: ret i32 %sext
}

define i32 @test16(i16 %x) nounwind {
  %and = and i16 %x, 8
  %cmp = icmp eq i16 %and, 8
  %ext = sext i1 %cmp to i32
  ret i32 %ext
; CHECK-LABEL: @test16(
; CHECK-NEXT: %1 = shl i16 %x, 12
; CHECK-NEXT: %sext = ashr i16 %1, 15
; CHECK-NEXT: %ext = sext i16 %sext to i32
; CHECK-NEXT: ret i32 %ext
}

define i32 @test17(i1 %x) nounwind {
  %c1 = sext i1 %x to i32
  %c2 = sub i32 0, %c1
  ret i32 %c2
; CHECK-LABEL: @test17(
; CHECK-NEXT: [[TEST17:%.*]] = zext i1 %x to i32
; CHECK-NEXT: ret i32 [[TEST17]]
}

define i32 @test18(i16 %x) {
  %cmp = icmp slt i16 %x, 0
  %sel = select i1 %cmp, i16 0, i16 %x
  %ext = sext i16 %sel to i32
  ret i32 %ext
; CHECK-LABEL: @test18(
; CHECK-NEXT: %[[cmp:.*]] = icmp slt i16 %x, 0
; CHECK-NEXT: %[[sel:.*]] = select i1 %[[cmp]], i16 0, i16 %x
; CHECK-NEXT: %[[ext:.*]] = zext i16 %[[sel]] to i32
; CHECK-NEXT: ret i32 %[[ext]]
}
