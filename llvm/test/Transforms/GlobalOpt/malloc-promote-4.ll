; RUN: opt -S -passes=globalopt -o - < %s | FileCheck %s

; CHECK: [[G_INIT:@.*]] = internal unnamed_addr global i1 false
@g = internal global i32* null, align 8

; CHECK-LABEL: define {{.*}} @f1(
; CHECK-NEXT:    [[G_INIT_VAL:%.*]] = load i1, i1* [[G_INIT]], align 1
; CHECK-NEXT:    call fastcc void @f2()
; CHECK-NEXT:    [[NOTINIT:%.*]] = xor i1 [[G_INIT_VAL]], true
; CHECK-NEXT:    br i1 [[NOTINIT]], label [[TMP1:%.*]], label [[TMP2:%.*]]
;
define internal i32 @f1() {
  %1 = load i32*, i32** @g, align 8
  call void @f2();
  %2 = icmp eq i32* %1, null
  br i1 %2, label %3, label %4

3:                                          ; preds = %0
  br label %5

4:                                          ; preds = %0
  br label %5

5:                                          ; preds = %3, %4
  %6 = phi i32 [ -1, %3 ], [ 1, %4 ]
  ret i32 %6
}

; CHECK-LABEL: define {{.*}} @f2(
; CHECK-NEXT:    store i1 true, i1* [[G_INIT]], align 1
; CHECK-NEXT:    ret void
;
define internal void @f2() {
  %1 = call noalias i8* @malloc(i64 4)
  %2 = bitcast i8* %1 to i32*
  store i32* %2, i32** @g, align 8
  ret void
}

; CHECK-LABEL: define {{.*}} @main(
; CHECK-NEXT:    store i1 false, i1* [[G_INIT]], align 1
; CHECK-NEXT:    [[TMP1:%.*]] = call fastcc i32 @f1()
; CHECK-NEXT:    ret i32 [[TMP1]]
;
define dso_local i32 @main() {
  store i32* null, i32** @g, align 8
  %1 = call i32 @f1()
  ret i32 %1
}

declare dso_local noalias i8* @malloc(i64)
