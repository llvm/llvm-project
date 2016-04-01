; RUN: opt -S -always-inline < %s | FileCheck %s

declare i8 @llvm.experimental.deoptimize.i8(...)

define i8 @callee(i1* %c) alwaysinline {
  %c0 = load volatile i1, i1* %c
  br i1 %c0, label %left, label %right

left:
  %c1 = load volatile i1, i1* %c
  br i1 %c1, label %lleft, label %lright

lleft:
  %v0 = call i8(...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i32 1) ]
  ret i8 %v0

lright:
  ret i8 10

right:
  %c2 = load volatile i1, i1* %c
  br i1 %c2, label %rleft, label %rright

rleft:
  %v1 = call i8(...) @llvm.experimental.deoptimize.i8(i32 1, i32 300, float 500.0, <2 x i32*> undef) [ "deopt"(i32 1) ]
  ret i8 %v1

rright:
  %v2 = call i8(...) @llvm.experimental.deoptimize.i8() [ "deopt"(i32 1) ]
  ret i8 %v2
}

define void @caller_0(i1* %c, i8* %ptr) {
; CHECK-LABEL: @caller_0(
entry:
  %v = call i8 @callee(i1* %c)  [ "deopt"(i32 2) ]
  store i8 %v, i8* %ptr
  ret void

; CHECK: lleft.i:
; CHECK-NEXT:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1) [ "deopt"(i32 2, i32 1) ]
; CHECK-NEXT:  ret void

; CHECK: rleft.i:
; CHECK-NEXT:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1, i32 300, float 5.000000e+02, <2 x i32*> undef) [ "deopt"(i32 2, i32 1) ]
; CHECK-NEXT:  ret void

; CHECK: rright.i:
; CHECK-NEXT:  call void (...) @llvm.experimental.deoptimize.isVoid() [ "deopt"(i32 2, i32 1) ]
; CHECK-NEXT:  ret void

; CHECK: callee.exit:
; CHECK-NEXT:  store i8 10, i8* %ptr
; CHECK-NEXT:  ret void

}

define i32 @caller_1(i1* %c, i8* %ptr) personality i8 3 {
; CHECK-LABEL: @caller_1(
entry:
  %v = invoke i8 @callee(i1* %c)  [ "deopt"(i32 3) ] to label %normal
       unwind label %unwind

; CHECK: lleft.i:
; CHECK-NEXT:  %0 = call i32 (...) @llvm.experimental.deoptimize.i32(i32 1) [ "deopt"(i32 3, i32 1) ]
; CHECK-NEXT:  ret i32 %0

; CHECK: rleft.i:
; CHECK-NEXT:  %1 = call i32 (...) @llvm.experimental.deoptimize.i32(i32 1, i32 300, float 5.000000e+02, <2 x i32*> undef) [ "deopt"(i32 3, i32 1) ]
; CHECK-NEXT:  ret i32 %1

; CHECK: rright.i:
; CHECK-NEXT:  %2 = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 3, i32 1) ]
; CHECK-NEXT:  ret i32 %2

; CHECK: callee.exit:
; CHECK-NEXT:  br label %normal

; CHECK: normal:
; CHECK-NEXT:  store i8 10, i8* %ptr
; CHECK-NEXT:  ret i32 42

unwind:
  %lp = landingpad i32 cleanup
  ret i32 43

normal:
  store i8 %v, i8* %ptr
  ret i32 42
}

define i8 @callee_with_alloca() alwaysinline {
  %t = alloca i8
  %v0 = call i8(...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i8* %t) ]
  ret i8 %v0
}

define void @caller_with_lifetime() {
; CHECK-LABLE: @caller_with_lifetime(
; CHECK:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1) [ "deopt"(i8* %t.i) ]
; CHECK-NEXT:  ret void

entry:
  call i8 @callee_with_alloca();
  ret void
}

define i8 @callee_with_dynamic_alloca(i32 %n) alwaysinline {
  %p = alloca i8, i32 %n
  %v = call i8(...) @llvm.experimental.deoptimize.i8(i32 1) [ "deopt"(i8* %p) ]
  ret i8 %v
}

define void @caller_with_stacksaverestore(i32 %n) {
; CHECK-LABEL: void @caller_with_stacksaverestore(
; CHECK:  call void (...) @llvm.experimental.deoptimize.isVoid(i32 1) [ "deopt"(i8* %p.i) ]
; CHECK-NEXT:  ret void

  %p = alloca i32, i32 %n
  call i8 @callee_with_dynamic_alloca(i32 %n)
  ret void
}
