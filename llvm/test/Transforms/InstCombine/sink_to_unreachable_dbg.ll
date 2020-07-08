; RUN: opt -debugify -debugify-level=locations -instcombine -S < %s | FileCheck %s

; CHECK-LABEL: @test1(
; CHECK: [[phi:%.*]] = phi i32
; CHECK-NEXT: [[add:%.*]] = add i32 {{.*}}, 1{{$}}
; CHECK-NEXT: add i32 [[phi]], [[add]], !dbg
define i32 @test1(i32 %0, i1 %1) {
  %3 = add i32 %0, 1
  br i1 %1, label %4, label %5

4:                                                ; preds = %2
  br label %6

5:                                                ; preds = %2
  br label %6

6:                                                ; preds = %5, %4
  %7 = phi i32 [ 0, %4 ], [ 1, %5 ]
  %8 = add i32 %7, %3
  ret i32 %8
}

; Function Attrs: nounwind readnone
declare i32 @external(i32) #0

; CHECK-LABEL: @test2(
; CHECK: [[phi:%.*]] = phi i32
; CHECK-NEXT: [[add:%.*]] = call i32 @external(i32 {{.*}}), !dbg
; CHECK-NEXT: add i32 [[phi]], [[add]], !dbg
define i32 @test2(i32 %0, i1 %1) {
  %3 = call i32 @external(i32 %0)
  br i1 %1, label %4, label %5

4:                                                ; preds = %2
  br label %6

5:                                                ; preds = %2
  br label %6

6:                                                ; preds = %5, %4
  %7 = phi i32 [ 0, %4 ], [ 1, %5 ]
  %8 = add i32 %7, %3
  ret i32 %8
}

attributes #0 = { nounwind readnone }
