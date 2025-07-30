; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare void @callee0()
declare void @callee1(i32,i32)

define void @f0(i32* %ptr) {
; CHECK-LABEL: @f0(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1
  call void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float  0.000000e+00, i64 100, i32 %l) ]
; CHECK: call void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float  0.000000e+00, i64 100, i32 %l) ]
  ret void
}

define void @f1(i32* %ptr) {
; CHECK-LABEL: @f1(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1

  call void @callee0()
  call void @callee0() [ "foo"() ]
  call void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float  0.000000e+00, i64 100, i32 %l) ]
; CHECK: @callee0(){{$}}
; CHECK-NEXT: call void @callee0() [ "foo"() ]
; CHECK-NEXT: call void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float  0.000000e+00, i64 100, i32 %l) ]
  ret void
}

define void @f2(i32* %ptr) {
; CHECK-LABEL: @f2(
 entry:
  call void @callee0() [ "foo"() ]
; CHECK: call void @callee0() [ "foo"() ]
  ret void
}

define void @f3(i32* %ptr) {
; CHECK-LABEL: @f3(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1
  call void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]
; CHECK: call void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]
  ret void
}

define void @f4(i32* %ptr) {
; CHECK-LABEL: @f4(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1
  call void @callee1(i32 10, i32 %x) [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]
; CHECK: call void @callee1(i32 10, i32 %x) [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]
  ret void
}

define void @f5(i32 %x) {
entry:
  call void @callee1(i32 10, i32 %x) [ "foo"(i32 42, metadata !"abc"), "bar"(metadata !"abcde", metadata !"qwerty") ]
; CHECK: call void @callee1(i32 10, i32 %x) [ "foo"(i32 42, metadata !"abc"), "bar"(metadata !"abcde", metadata !"qwerty") ]
  ret void
}

; Invoke versions of the above tests:


define void @g0(i32* %ptr) personality i8 3 {
; CHECK-LABEL: @g0(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1
  invoke void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float  0.000000e+00, i64 100, i32 %l) ] to label %normal unwind label %exception
; CHECK: invoke void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float  0.000000e+00, i64 100, i32 %l) ]

exception:
  %cleanup = landingpad i8 cleanup
  br label %normal
normal:
  ret void
}

define void @g1(i32* %ptr) personality i8 3 {
; CHECK-LABEL: @g1(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1

  invoke void @callee0() to label %normal unwind label %exception
; CHECK: invoke void @callee0(){{$}}

exception:
  %cleanup = landingpad i8 cleanup
  br label %normal

normal:
  invoke void @callee0() [ "foo"() ] to label %normal1 unwind label %exception1
; CHECK: invoke void @callee0() [ "foo"() ]

exception1:
  %cleanup1 = landingpad i8 cleanup
  br label %normal1

normal1:
  invoke void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ] to label %normal2 unwind label %exception2
; CHECK: invoke void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]

exception2:
  %cleanup2 = landingpad i8 cleanup
  br label %normal2

normal2:
  ret void
}

define void @g2(i32* %ptr) personality i8 3 {
; CHECK-LABEL: @g2(
 entry:
  invoke void @callee0() [ "foo"() ] to label %normal unwind label %exception
; CHECK: invoke void @callee0() [ "foo"() ]

exception:
  %cleanup = landingpad i8 cleanup
  br label %normal
normal:
  ret void
}

define void @g3(i32* %ptr) personality i8 3 {
; CHECK-LABEL: @g3(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1
  invoke void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ] to label %normal unwind label %exception
; CHECK: invoke void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]

exception:
  %cleanup = landingpad i8 cleanup
  br label %normal
normal:
  ret void
}

define void @g4(i32* %ptr) personality i8 3 {
; CHECK-LABEL: @g4(
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1
  invoke void @callee1(i32 10, i32 %x) [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]
        to label %normal unwind label %exception
; CHECK: invoke void @callee1(i32 10, i32 %x) [ "foo"(i32 42, i64 100, i32 %x), "foo"(i32 42, float  0.000000e+00, i32 %l) ]

exception:
  %cleanup = landingpad i8 cleanup
  br label %normal
normal:
  ret void
}

define void @g5(ptr %ptr) personality i8 3 {
entry:
  %l = load i32, ptr %ptr, align 4
  %x = add i32 42, 1
  invoke void @callee1(i32 10, i32 %x) [ "foo"(i32 42, metadata !"abc"), "bar"(metadata !"abcde", metadata !"qwerty") ]
          to label %normal unwind label %exception
; CHECK:   invoke void @callee1(i32 10, i32 %x) [ "foo"(i32 42, metadata !"abc"), "bar"(metadata !"abcde", metadata !"qwerty") ]

exception:                                        ; preds = %entry
  %cleanup = landingpad i8
          cleanup
  br label %normal

normal:                                           ; preds = %exception, %entry
  ret void
}
