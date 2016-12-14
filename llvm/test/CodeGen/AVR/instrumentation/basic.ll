; RUN: opt -S -avr-instrument-functions < %s | FileCheck %s

; Functions returning void should not be instrumented.
; CHECK-LABEL: do_nothing
define void @do_nothing(i8 %c) {
  ; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: do_something
define i8 @do_something(i16 %a, i16 %b) {
  ; CHECK: instrumentation_entry
  ; CHECK-NEXT: %0 = getelementptr inbounds [13 x i8], [13 x i8]* @0, i8 0, i8 0
  ; CHECK-NEXT: call void @avr_instrumentation_begin_signature(i8* %0, i16 2)

  ; CHECK-NEXT: %1 = getelementptr inbounds [2 x i8], [2 x i8]* @1, i8 0, i8 0
  ; CHECK-NEXT: call void @avr_instrumentation_argument_i16(i8* %1, i16 %a)

  ; CHECK-NEXT: %2 = getelementptr inbounds [2 x i8], [2 x i8]* @2, i8 0, i8 0
  ; CHECK-NEXT: call void @avr_instrumentation_argument_i16(i8* %2, i16 %b)

  ; CHECK-NEXT: %3 = getelementptr inbounds [13 x i8], [13 x i8]* @3, i8 0, i8 0
  ; CHECK-NEXT: call void @avr_instrumentation_end_signature(i8* %3, i16 2)

  ; CHECK-NEXT: br label %4

  ; CHECK: call void @avr_instrumentation_result_u8(i8 1)
  ; CHECK-NEXT: ret i8 1
  ret i8 1
}

; CHECK-LABEL: foo
define i32 @foo() {
  ; CHECK: instrumentation_entry:
  ; CHECK-NEXT:   %0 = getelementptr inbounds [4 x i8], [4 x i8]* @4, i8 0, i8 0
  ; CHECK-NEXT:   call void @avr_instrumentation_begin_signature(i8* %0, i16 0)
  ; CHECK-NEXT:   %1 = getelementptr inbounds [4 x i8], [4 x i8]* @5, i8 0, i8 0
  ; CHECK-NEXT:   call void @avr_instrumentation_end_signature(i8* %1, i16 0)

  ; CHECK-NEXT:   br label %2

  ; CHECK:         call void @avr_instrumentation_result_u32(i32 50)
  ; CHECK-NEXT:   ret i32 50
  ret i32 50
}
