; RUN: opt -passes=attributor-cgscc  -attributor-annotate-decl-cs -attributor-allow-shallow-wrappers -S < %s | FileCheck %s --check-prefix=CHECK

; TEST 1: simple test, without argument
; A wrapper will be generated for this function, Check the wrapper first
; CHECK-NOT: Function Attrs:
; CHECK: define linkonce i32 @inner1()
; CHECK: tail call i32 @0()
; CHECK: ret
;
; Check the original function, which is wrapped and becomes anonymous
; CHECK: Function Attrs: nofree norecurse nosync nounwind readnone willreturn
; CHECK: define internal i32 @0()
; CHECK: ret i32 1
define linkonce i32 @inner1() {
entry:
  %a = alloca i32
  store i32 1, i32* %a
  %b = load i32, i32* %a
  ret i32 %b
}

; Check for call
; CHECK: define i32 @outer1
; CHECK: call i32 @inner1
; CHECK: ret
define i32 @outer1() {
entry:
  %ret = call i32 @inner1()
  ret i32 %ret
}

; TEST 2: with argument
; CHECK-NOT: Function Attrs
; CHECK: define linkonce i32 @inner2(i32 %a, i32 %b)
; CHECK: tail call i32 @1(i32 %a, i32 %b)
; CHECK: ret
;
; CHECK: Function Attrs: nofree norecurse nosync nounwind readnone willreturn
; CHECK: define internal i32 @1(i32 %a, i32 %b)
; CHECK: %c = add i32 %a, %b
; CHECK: ret i32 %c
define linkonce i32 @inner2(i32 %a, i32 %b) {
entry:
  %c = add i32 %a, %b
  ret i32 %c
}

; CHECK: define i32 @outer2
; CHECK: call i32 @inner2
; CHECK: ret
define i32 @outer2() {
entry:
  %ret = call i32 @inner2(i32 1, i32 2)
  ret i32 %ret
}

; TEST 3: check nocurse
; This function calls itself, there will be no attribute
; CHECK-NOT: Function Attrs
; CHECK: define linkonce i32 @inner3(i32 %0)
; CHECK: tail call i32 @2(i32 %0)
; CHECK: ret
;
; CHECK-NOT: Function Attrs:
; CHECK: define internal i32 @2(i32 %0)
define linkonce i32 @inner3(i32) {
entry:
  %1 = alloca i32
  store i32 %0, i32* %1
  br label %2
2:
  %3 = load i32, i32* %1
  %4 = icmp slt i32 %3, 4
  br i1 %4, label %5, label %9
5:
  %6 = load i32, i32* %1
  %7 = add nsw i32 %6, 1
  %8 = call i32 @inner3(i32 %7)
  store i32 %8, i32* %1
  br label %2
9:
  %10 = load i32, i32* %1
  ret i32 %10
}

