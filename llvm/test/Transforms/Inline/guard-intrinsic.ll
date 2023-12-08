; RUN: opt -S -passes=always-inline < %s | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define i8 @callee(ptr %c_ptr) alwaysinline {
  %c = load volatile i1, ptr %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c, i32 1) [ "deopt"(i32 1) ]
  ret i8 5
}

define void @caller_0(ptr %c, ptr %ptr) {
; CHECK-LABEL: @caller_0(
entry:
; CHECK:  [[COND:%[^ ]+]] = load volatile i1, ptr %c
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 [[COND]], i32 1) [ "deopt"(i32 2, i32 1) ]
; CHECK-NEXT:  store i8 5, ptr %ptr

  %v = call i8 @callee(ptr %c)  [ "deopt"(i32 2) ]
  store i8 %v, ptr %ptr
  ret void
}

define i32 @caller_1(ptr %c, ptr %ptr) personality i8 3 {
; CHECK-LABEL: @caller_1(
; CHECK:  [[COND:%[^ ]+]] = load volatile i1, ptr %c
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 [[COND]], i32 1) [ "deopt"(i32 3, i32 1) ]
; CHECK-NEXT:  br label %normal
entry:
  %v = invoke i8 @callee(ptr %c)  [ "deopt"(i32 3) ] to label %normal
       unwind label %unwind

unwind:
  %lp = landingpad i32 cleanup
  ret i32 43

normal:
  store i8 %v, ptr %ptr
  ret i32 42
}
