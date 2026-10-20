; RUN: opt -passes=mergefunc -S < %s | FileCheck %s

define i8 @call_with_range() {
  bitcast i8 0 to i8 ; dummy to make the function large enough
  %out = call i8 @dummy(), !range !0
  ret i8 %out
}

define i8 @call_no_range() {
; CHECK-LABEL: @call_no_range
; CHECK-NEXT: bitcast i8 0 to i8
; CHECK-NEXT: %out = call i8 @dummy()
; CHECK-NEXT: ret i8 %out
  bitcast i8 0 to i8
  %out = call i8 @dummy()
  ret i8 %out
}

define i8 @call_different_range() {
; CHECK-LABEL: @call_different_range
; CHECK-NEXT: bitcast i8 0 to i8
; CHECK-NEXT: %out = call i8 @dummy(), !range !1
; CHECK-NEXT: ret i8 %out
  bitcast i8 0 to i8
  %out = call i8 @dummy(), !range !1
  ret i8 %out
}

define i8 @invoke_with_range() personality ptr undef {
  %out = invoke i8 @dummy() to label %next unwind label %lpad, !range !0

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
; CHECK-NEXT: invoke i8 @dummy
  %out = invoke i8 @dummy() to label %next unwind label %lpad, !range !1

next:
  ret i8 %out

lpad:
  %pad = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } zeroinitializer
}

define i8 @invoke_with_same_range() personality ptr undef {
; CHECK-DAG: @invoke_with_same_range()
; CHECK-DAG: tail call i8 @invoke_with_range()
  %out = invoke i8 @dummy() to label %next unwind label %lpad, !range !0

next:
  ret i8 %out

lpad:
  %pad = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } zeroinitializer
}

define i8 @call_with_same_range() {
; CHECK-DAG: @call_with_same_range
; CHECK-DAG: tail call i8 @call_with_range
  bitcast i8 0 to i8
  %out = call i8 @dummy(), !range !0
  ret i8 %out
}


declare i8 @dummy();
declare i32 @__gxx_personality_v0(...)

!0 = !{i8 0, i8 2}
!1 = !{i8 5, i8 7}
