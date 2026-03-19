; RUN: opt -passes=mergefunc -S < %s | FileCheck %s

define i8 @call_with_range_attr(i8 range(i8 0, 2) %v) {
  %out = call i8 @dummy2(i8 %v)
  ret i8 %out
}

define i8 @call_no_range_attr(i8 %v) {
; CHECK-LABEL: @call_no_range_attr
; CHECK-NEXT: %out = call i8 @dummy2(i8 %v)
; CHECK-NEXT: ret i8 %out
  %out = call i8 @dummy2(i8 %v)
  ret i8 %out
}

define i8 @call_different_range_attr(i8 range(i8 5, 7) %v) {
; CHECK-LABEL: @call_different_range_attr
; CHECK-NEXT: %out = call i8 @dummy2(i8 %v)
; CHECK-NEXT: ret i8 %out
  %out = call i8 @dummy2(i8 %v)
  ret i8 %out
}

define i8 @call_with_range() {
  %out = call range(i8 0, 2) i8 @dummy()
  ret i8 %out
}

define i8 @call_no_range() {
; CHECK-LABEL: @call_no_range
; CHECK-NEXT: %out = call i8 @dummy()
; CHECK-NEXT: ret i8 %out
  %out = call i8 @dummy()
  ret i8 %out
}

define i8 @call_different_range() {
; CHECK-LABEL: @call_different_range
; CHECK-NEXT: %out = call range(i8 5, 7) i8 @dummy()
; CHECK-NEXT: ret i8 %out
  %out = call range(i8 5, 7) i8 @dummy()
  ret i8 %out
}

define i8 @invoke_with_range() personality ptr undef {
  %out = invoke range(i8 0, 2) i8 @dummy() to label %next unwind label %lpad

next:
  ret i8 %out

lpad:
  %pad = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } zeroinitializer
}

define i8 @invoke_no_range() personality ptr undef {
; CHECK-LABEL: @invoke_no_range()
; CHECK-NEXT: invoke i8 @dummy
  %out = invoke i8 @dummy() to label %next unwind label %lpad

next:
  ret i8 %out

lpad:
  %pad = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } zeroinitializer
}

define i8 @invoke_different_range() personality ptr undef {
; CHECK-LABEL: @invoke_different_range()
; CHECK-NEXT: invoke range(i8 5, 7) i8 @dummy
  %out = invoke range(i8 5, 7) i8 @dummy() to label %next unwind label %lpad

next:
  ret i8 %out

lpad:
  %pad = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } zeroinitializer
}

define i8 @invoke_with_same_range() personality ptr undef {
; CHECK-DAG: @invoke_with_same_range()
; CHECK-DAG: tail call i8 @invoke_with_range()
  %out = invoke range(i8 0, 2) i8 @dummy() to label %next unwind label %lpad

next:
  ret i8 %out

lpad:
  %pad = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } zeroinitializer
}

define i8 @call_with_same_range() {
; CHECK-DAG: @call_with_same_range()
; CHECK-DAG: tail call i8 @call_with_range()
  %out = call range(i8 0, 2) i8 @dummy()
  ret i8 %out
}

define i8 @call_with_same_range_attr(i8 range(i8 0, 2) %v) {
; CHECK-DAG: @call_with_same_range_attr
; CHECK-DAG: tail call i8 @call_with_range_attr
  %out = call i8 @dummy2(i8 %v)
  ret i8 %out
}

declare i8 @dummy();
declare i8 @dummy2(i8);
declare i32 @__gxx_personality_v0(...)
