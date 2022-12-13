; RUN: opt -S --passes=ipsccp,deadargelim -specialize-functions -force-function-specialization < %s | FileCheck %s
define dso_local i32 @add(i32 %x, i32 %y) {
entry:
  %add = add nsw i32 %y, %x
  ret i32 %add
}

define dso_local i32 @sub(i32 %x, i32 %y) {
entry:
  %sub = sub nsw i32 %x, %y
  ret i32 %sub
}

define internal i32 @f(i32 %x, i32 %y, ptr %u, ptr %v) noinline {
entry:
  %call = tail call i32 %u(i32 %x, i32 %y)
  %call1 = tail call i32 %v(i32 %x, i32 %y)
  %mul = mul nsw i32 %call1, %call
  ret i32 %mul
}

define dso_local i32 @g0(i32 %x, i32 %y) {
; CHECK-LABEL: @g0
; CHECK:       call i32 @f.2(i32 [[X:%.*]], i32 [[Y:%.*]])
entry:
  %call = tail call i32 @f(i32 %x, i32 %y, ptr @add, ptr @add)
  ret i32 %call
}


define dso_local i32 @g1(i32 %x, i32 %y) {
; CHECK-LABEL: @g1(
; CHECK:       call i32 @f.1(i32 [[X:%.*]], i32 [[Y:%.*]])
entry:
  %call = tail call i32 @f(i32 %x, i32 %y, ptr @sub, ptr @add)
  ret i32 %call
}

define dso_local i32 @g2(i32 %x, i32 %y, ptr %v) {
; CHECK-LABEL @g2
; CHECK       call i32 @f.3(i32 [[X:%.*]], i32 [[Y:%.*]], ptr [[V:%.*]])
entry:
  %call = tail call i32 @f(i32 %x, i32 %y, ptr @sub, ptr %v)
  ret i32 %call
}

; CHECK-LABEL: define {{.*}} i32 @f.1
; CHECK:       call i32 @sub(i32 %x, i32 %y)
; CHECK-NEXT:  call i32 @add(i32 %x, i32 %y)

; CHECK-LABEL: define {{.*}} i32 @f.2
; CHECK:       call i32 @add(i32 %x, i32 %y)
; CHECK-NEXT   call i32 @add(i32 %x, i32 %y)

; CHECK-LABEL: define {{.*}} i32 @f.3
; CHECK:       call i32 @sub(i32 %x, i32 %y)
; CHECK-NEXT:  call i32 %v(i32 %x, i32 %y)

