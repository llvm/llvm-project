; RUN: opt < %s -passes=aggressive-instcombine -mtriple=x86_64 -S | FileCheck %s

; This LIT test checks if TruncInstCombine pass correctly recognizes the
; constraints from a signed min-max clamp. The clamp is a sequence of smin and
; smax instructions limiting a variable into a range, smin <= x <= smax.

declare i16 @llvm.smin.i16(i16, i16)
declare i16 @llvm.smax.i16(i16, i16)


; CHECK-LABEL: @test_1
; CHECK-NEXT: [[ONE:%.*]] = tail call i16 @llvm.smin.i16(i16 [[X:%.*]], i16 31)
; CHECK-NEXT: [[TWO:%.*]] = tail call i16 @llvm.smax.i16(i16 [[ONE]], i16 0)
; CHECK-NEXT: [[A:%.*]] = trunc i16 [[TWO]] to i8
; CHECK-NEXT: [[B:%.*]] = lshr i8 [[A]], 2
; CHECK-NEXT: ret i8 [[B]]

define i8 @test_1(i16 %x) {
  %1 = tail call i16 @llvm.smin.i16(i16 %x, i16 31)
  %2 = tail call i16 @llvm.smax.i16(i16 %1, i16 0)
  %a = sext i16 %2 to i32
  %b = lshr i32 %a, 2
  %b.trunc = trunc i32 %b to i8
  ret i8 %b.trunc
}


; CHECK-LABEL: @test_1a
; CHECK-NEXT: [[ONE:%.*]] = tail call i16 @llvm.smin.i16(i16 [[X:%.*]], i16 31)
; CHECK-NEXT: [[TWO:%.*]] = tail call i16 @llvm.smax.i16(i16 [[ONE]], i16 0)
; CHECK-NEXT: [[A:%.*]] = trunc i16 [[TWO]] to i8
; CHECK-NEXT: [[B:%.*]] = add i8 [[A]], 2
; CHECK-NEXT: ret i8 [[B]]

define i8 @test_1a(i16 %x) {
  %1 = tail call i16 @llvm.smin.i16(i16 %x, i16 31)
  %2 = tail call i16 @llvm.smax.i16(i16 %1, i16 0)
  %a = sext i16 %2 to i32
  %b = add i32 %a, 2
  %b.trunc = trunc i32 %b to i8
  ret i8 %b.trunc
}


; CHECK-LABEL: @test_2
; CHECK-NEXT: [[ONE:%.*]] = tail call i16 @llvm.smin.i16(i16 [[X:%.*]], i16 -1)
; CHECK-NEXT: [[TWO:%.*]] = tail call i16 @llvm.smax.i16(i16 [[ONE]], i16 -31)
; CHECK-NEXT: [[A:%.*]] = trunc i16 [[TWO]] to i8
; CHECK-NEXT: [[B:%.*]] = add i8 [[A]], 2
; CHECK-NEXT: ret i8 [[B]]

define i8 @test_2(i16 %x) {
  %1 = tail call i16 @llvm.smin.i16(i16 %x, i16 -1)
  %2 = tail call i16 @llvm.smax.i16(i16 %1, i16 -31)
  %a = sext i16 %2 to i32
  %b = add i32 %a, 2
  %b.trunc = trunc i32 %b to i8
  ret i8 %b.trunc
}


; CHECK-LABEL: @test_3
; CHECK-NEXT: [[ONE:%.*]] = tail call i16 @llvm.smin.i16(i16 [[X:%.*]], i16 31)
; CHECK-NEXT: [[TWO:%.*]] = tail call i16 @llvm.smax.i16(i16 [[ONE]], i16 -31)
; CHECK-NEXT: [[A:%.*]] = trunc i16 [[TWO]] to i8
; CHECK-NEXT: [[B:%.*]] = add i8 [[A]], 2
; CHECK-NEXT: ret i8 [[B]]

define i8 @test_3(i16 %x) {
  %1 = tail call i16 @llvm.smin.i16(i16 %x, i16 31)
  %2 = tail call i16 @llvm.smax.i16(i16 %1, i16 -31)
  %a = sext i16 %2 to i32
  %b = add i32 %a, 2
  %b.trunc = trunc i32 %b to i8
  ret i8 %b.trunc
}


; CHECK-LABEL: @test_4
; CHECK-NEXT: [[ONE:%.*]] = tail call i16 @llvm.smin.i16(i16 [[X:%.*]], i16 127)
; CHECK-NEXT: [[TWO:%.*]] = tail call i16 @llvm.smax.i16(i16 [[ONE]], i16 0)
; CHECK-NEXT: [[THREE:%.*]] = tail call i16 @llvm.smin.i16(i16 [[Y:%.*]], i16 127)
; CHECK-NEXT: [[FOUR:%.*]] = tail call i16 @llvm.smax.i16(i16 [[THREE]], i16 0)
; CHECK-NEXT: [[A:%.*]] = mul i16 [[TWO]], [[FOUR]]
; CHECK-NEXT: [[B:%.*]] = lshr i16 [[A]], 7
; CHECK-NEXT: [[C:%.*]] = trunc i16 [[B]] to i8
; CHECK-NEXT: ret i8 [[C]]

define i8 @test_4(i16 %x, i16 %y) {
  %1 = tail call i16 @llvm.smin.i16(i16 %x, i16 127)
  %2 = tail call i16 @llvm.smax.i16(i16 %1, i16 0)
  %x.clamp = zext nneg i16 %2 to i32
  %3 = tail call i16 @llvm.smin.i16(i16 %y, i16 127)
  %4 = tail call i16 @llvm.smax.i16(i16 %3, i16 0)
  %y.clamp = zext nneg i16 %4 to i32
  %mul = mul nuw nsw i32 %x.clamp, %y.clamp
  %shr = lshr i32 %mul, 7
  %trunc= trunc nuw nsw i32 %shr to i8
  ret i8 %trunc
}
