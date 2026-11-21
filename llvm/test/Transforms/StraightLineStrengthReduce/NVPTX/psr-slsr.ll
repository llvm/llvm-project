; RUN: opt < %s -passes=slsr -enable-partial-strength-reduce=0 -psr-threshold=0 -S | FileCheck %s -check-prefixes=CHECK,CHECK-SLSR
; RUN: opt < %s -passes=slsr -enable-partial-strength-reduce -psr-threshold=1 -S | FileCheck %s -check-prefixes=CHECK,CHECK-PSR

; These tests can be optimized by either SLSR or PSR, and PSR gives better output

declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; CHECK-LABEL: test1
; CHECK: entry:
; CHECK: [[GEP1:%.*]] = getelementptr i32, ptr %a, i64 0

; CHECK-SLSR: true:
; CHECK-SLSR: [[I1:%.*]] = zext i32 %0 to i64
; CHECK-SLSR: [[INDEX1:%.*]] = shl i64 [[I1]], 2
; CHECK-SLSR: [[GEP2:%.*]] = getelementptr i8, ptr [[GEP1]], i64 [[INDEX1]]
; CHECK-SLSR: store i32 0, ptr [[GEP2]], align 4
; CHECK-SLSR: false:
; CHECK-SLSR: [[I2:%.*]] = zext i32 %0 to i64
; CHECK-SLSR: [[INDEX2:%.*]] = shl i64 [[I2]], 2
; CHECK-SLSR: [[GEP3:%.*]] = getelementptr i8, ptr [[GEP1]], i64 [[INDEX2]]
; CHECK-SLSR: store i32 1, ptr [[GEP3]], align 4

; CHECK-PSR: [[I:%.*]] = zext i32 %0 to i64
; CHECK-PSR: [[INDEX:%.*]] = shl i64 [[I]], 2
; CHECK-PSR: [[GEP2:%.*]] = getelementptr i8, ptr [[GEP1]], i64 [[INDEX]]
; CHECK-PSR: true:
; CHECK-PSR: store i32 0, ptr [[GEP2]], align 4
; CHECK-PSR: false:
; CHECK-PSR: store i32 1, ptr [[GEP2]], align 4
define void @test1(i32 %0, ptr %a) {
entry:
  %1 = getelementptr i32, ptr %a, i64 0
  store i32 %0, ptr %1
  br i1 false, label %true, label %false

true:
  %2 = zext i32 %0 to i64
  %3 = getelementptr i32, ptr %a, i64 %2
  store i32 0, ptr %3
  br label %false

false:
  %4 = zext i32 %0 to i64
  %5 = getelementptr i32, ptr %a, i64 %4
  store i32 1, ptr %5
  ret void
}

; CHECK-LABEL: test2
; CHECK: entry:
; CHECK: [[V:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK: [[GEP1:%.*]] = getelementptr i32, ptr %a, i32 0

; CHECK-SLSR: true:
; CHECK-SLSR: [[I1:%.*]] = add i32 [[V]], 0
; CHECK-SLSR: [[GEP2:%.*]] = getelementptr i32, ptr %a, i32 [[I1]]
; CHECK-SLSR: store i32 1, ptr [[GEP2]], align 4
; CHECK-SLSR: false:
; CHECK-SLSR: [[I2:%.*]] = add i32 [[V]], 0
; CHECK-SLSR: [[GEP3:%.*]] = getelementptr i32, ptr %a, i32 [[I2]]
; CHECK-SLSR: store i32 2, ptr [[GEP3]], align 4

; CHECK-PSR: true:
; CHECK-PSR: [[I1:%.*]] = add i32 [[V]], 0
; CHECK-PSR: [[GEP2:%.*]] = getelementptr i32, ptr %a, i32 [[I1]]
; CHECK-PSR: store i32 1, ptr [[GEP2]], align 4
; CHECK-PSR: false:
; CHECK-PSR: [[I2:%.*]] = add i32 [[V]], 0
; CHECK-PSR: [[GEP3:%.*]] = getelementptr i32, ptr %a, i32 [[I2]]
; CHECK-PSR: store i32 2, ptr [[GEP3]], align 4

define void @test2(ptr %a) {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = getelementptr i32, ptr %a, i32 0
  store i32 0, ptr %1
  br i1 false, label %true, label %false

true:
  %2 = add i32 %0, 0
  %3 = getelementptr i32, ptr %a, i32 %2
  store i32 1, ptr %3
  br label %false

false:
  %5 = add i32 %0, 0
  %6 = getelementptr i32, ptr %a, i32 %5
  store i32 2, ptr %6

  ret void
}

; CHECK-LABEL: test3
; CHECK: entry:
; CHECK: [[V:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK: [[GEP1:%.*]] = getelementptr i32, ptr %a, i64 0

; CHECK-SLSR: true:
; CHECK-SLSR: [[I1:%.*]] = zext i32 [[V]] to i64
; CHECK-SLSR: [[INDEX1:%.*]] = shl i64 [[I1]], 2
; CHECK-SLSR: [[GEP2:%.*]] = getelementptr i8, ptr [[GEP1]], i64 [[INDEX1]]
; CHECK-SLSR: store i32 1, ptr [[GEP2]], align 4
; CHECK-SLSR: false:
; CHECK-SLSR: [[I2:%.*]] = zext i32 [[V]] to i64
; CHECK-SLSR: [[INDEX2:%.*]] = shl i64 [[I2]], 2
; CHECK-SLSR: [[GEP3:%.*]] = getelementptr i8, ptr [[GEP1]], i64 [[INDEX2]]
; CHECK-SLSR: store i32 2, ptr [[GEP3]], align 4
; CHECK-SLSR: store i32 3, ptr [[GEP3]], align 4
; CHECK-SLSR: store i32 4, ptr [[GEP3]], align 4

; CHECK-PSR: [[I:%.*]] = zext i32 [[V]] to i64
; CHECK-PSR: [[INDEX:%.*]] = shl i64 [[I]], 2
; CHECK-PSR: [[GEP2:%.*]] = getelementptr i8, ptr [[GEP1]], i64 [[INDEX]]
; CHECK-PSR: true:
; CHECK-PSR: store i32 1, ptr [[GEP2]], align 4
; CHECK-PSR: false:
; CHECK-PSR: store i32 2, ptr [[GEP2]], align 4
; CHECK-PSR: store i32 3, ptr [[GEP2]], align 4
; CHECK-PSR: store i32 4, ptr [[GEP2]], align 4

define void @test3(ptr %a) {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = getelementptr i32, ptr %a, i64 0
  store i32 0, ptr %1
  br i1 false, label %true, label %false

true:
  %2 = zext i32 %0 to i64
  %3 = getelementptr i32, ptr %a, i64 %2
  store i32 1, ptr %3
  br label %false

false:
  %5 = zext i32 %0 to i64
  %6 = getelementptr i32, ptr %a, i64 %5
  store i32 2, ptr %6
  %7 = zext i32 %0 to i64
  %8 = getelementptr i32, ptr %a, i64 %7
  store i32 3, ptr %8
  %9 = zext i32 %0 to i64
  %10 = getelementptr i32, ptr %a, i64 %9
  store i32 4, ptr %10
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
