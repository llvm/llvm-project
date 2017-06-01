; RUN: opt -S -always-inline < %s | FileCheck %s

declare void @f()
declare i32 @g()
declare fastcc i32 @g.fastcc()

define i32 @callee_0() alwaysinline {
 entry:
  call void @f()
  ret i32 2
}

define i32 @caller_0() {
; CHECK-LABEL: @caller_0(
 entry:
; CHECK: entry:
; CHECK-NEXT: call void @f()
; CHECK-NEXT: ret i32 2
  %x = call i32 @callee_0() [ "deopt"(i32 5) ]
  ret i32 %x
}

define i32 @callee_1() alwaysinline {
 entry:
  call void @f() [ "deopt"() ]
  call void @f() [ "deopt"(i32 0, i32 1) ]
  call void @f() [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ]
  ret i32 2
}

define i32 @caller_1() {
; CHECK-LABEL: @caller_1(
 entry:
; CHECK: entry:
; CHECK-NEXT:  call void @f() [ "deopt"(i32 5) ]
; CHECK-NEXT:  call void @f() [ "deopt"(i32 5, i32 0, i32 1) ]
; CHECK-NEXT:  call void @f() [ "deopt"(i32 5, i32 0, i32 1), "foo"(double 0.000000e+00) ]
; CHECK-NEXT:  ret i32 2

  %x = call i32 @callee_1() [ "deopt"(i32 5) ]
  ret i32 %x
}

define i32 @callee_2() alwaysinline {
 entry:
  %v = call i32 @g() [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ]
  ret i32 %v
}

define i32 @caller_2(i32 %val) {
; CHECK-LABEL: @caller_2(
 entry:
; CHECK: entry:
; CHECK-NEXT:   [[RVAL:%[^ ]+]] = call i32 @g() [ "deopt"(i32 %val, i32 0, i32 1), "foo"(double 0.000000e+00) ]
; CHECK-NEXT:   ret i32 [[RVAL]]
  %x = call i32 @callee_2() [ "deopt"(i32 %val) ]
  ret i32 %x
}

define i32 @callee_3() alwaysinline {
 entry:
  %v = call i32 @g() [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ]
  ret i32 %v
}

define i32 @caller_3() personality i8 3 {
; CHECK-LABEL: @caller_3(
 entry:
  %x = invoke i32 @callee_3() [ "deopt"(i32 7) ] to label %normal unwind label %unwind
; CHECK: invoke i32 @g() [ "deopt"(i32 7, i32 0, i32 1), "foo"(double 0.000000e+00) ]

 normal:
  ret i32 %x

 unwind:
  %cleanup = landingpad i8 cleanup
  ret i32 101
}

define i32 @callee_4() alwaysinline personality i8 3 {
 entry:
  %v = invoke i32 @g() [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ] to label %normal unwind label %unwind

 normal:
  ret i32 %v

 unwind:
  %cleanup = landingpad i8 cleanup
  ret i32 100
}

define i32 @caller_4() {
; CHECK-LABEL: @caller_4(
 entry:
; CHECK: invoke i32 @g() [ "deopt"(i32 7, i32 0, i32 1), "foo"(double 0.000000e+00) ]
  %x = call i32 @callee_4() [ "deopt"(i32 7) ]
  ret i32 %x
}

define i32 @callee_5() alwaysinline personality i8 3 {
 entry:
  %v = invoke fastcc i32 @g.fastcc() #0 [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ] to label %normal unwind label %unwind

 normal:
  ret i32 %v

 unwind:
  %cleanup = landingpad i8 cleanup
  ret i32 100
}

define i32 @caller_5() {
; CHECK-LABEL: @caller_5(
 entry:
; CHECK:  invoke fastcc i32 @g.fastcc() #[[FOO_BAR_ATTR_IDX:[0-9]+]] [ "deopt"(i32 7, i32 0, i32 1), "foo"(double 0.000000e+00) ]
  %x = call i32 @callee_5() [ "deopt"(i32 7) ]
  ret i32 %x
}

define i32 @callee_6() alwaysinline personality i8 3 {
 entry:
  %v = call fastcc i32 @g.fastcc() #0 [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ]
  ret i32 %v
}

define i32 @caller_6() {
; CHECK-LABEL: @caller_6(
 entry:
; CHECK: call fastcc i32 @g.fastcc() #[[FOO_BAR_ATTR_IDX]] [ "deopt"(i32 7, i32 0, i32 1), "foo"(double 0.000000e+00) ]
  %x = call i32 @callee_6() [ "deopt"(i32 7) ]
  ret i32 %x
}

define i32 @callee_7(i1 %val) alwaysinline personality i8 3 {
; We want something that PruningFunctionCloner is not smart enough to
; recognize, but can be recognized by recursivelySimplifyInstruction.

 entry:
  br i1 %val, label %check, label %precheck

 precheck:
  br label %check

 check:
  %p = phi i1 [ %val, %entry ], [ true, %precheck ]
  br i1 %p, label %do.not, label %do

 do.not:
  ret i32 0

 do:
  %v = call fastcc i32 @g.fastcc() [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ]
  ret i32 %v
}

define i32 @caller_7() {
; CHECK-LABEL: @caller_7(
 entry:
; CHECK-NOT: call fastcc i32 @g.fastcc()
; CHECK: ret i32 0
  %x = call i32 @callee_7(i1 true) [ "deopt"(i32 7) ]
  ret i32 %x
}

define i32 @callee_8(i1 %val) alwaysinline personality i8 3 {
; We want something that PruningFunctionCloner is not smart enough to
; recognize, but can be recognized by recursivelySimplifyInstruction.

 entry:
  br i1 %val, label %check, label %precheck

 precheck:
  br label %check

 check:
  %p = phi i1 [ %val, %entry ], [ true, %precheck ]
  br i1 %p, label %do.not, label %do

 do.not:
  ret i32 0

 do:
  %phi = phi i32 [ 0, %check ], [ %v, %do ]
  %v = call fastcc i32 @g.fastcc() [ "deopt"(i32 0, i32 1), "foo"(double 0.0) ]
  %ic = icmp eq i32 %v, 42
  br i1 %ic, label %do, label %done

 done:
  ret i32 %phi
}

define i32 @caller_8() {
; CHECK-LABEL: @caller_8(
 entry:
; CHECK-NOT: call fastcc i32 @g.fastcc()
; CHECK: ret i32 0
  %x = call i32 @callee_8(i1 true) [ "deopt"(i32 7) ]
  ret i32 %x
}

attributes #0 = { "foo"="bar" }

; CHECK: attributes #[[FOO_BAR_ATTR_IDX]] = { "foo"="bar" }
