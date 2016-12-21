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
  ; CHECK-NEXT: call void @avr_instrumentation_argument_i16(i8* %1, i8 0, i16 %a)

  ; CHECK-NEXT: %2 = getelementptr inbounds [2 x i8], [2 x i8]* @2, i8 0, i8 0
  ; CHECK-NEXT: call void @avr_instrumentation_argument_i16(i8* %2, i8 1, i16 %b)

  ; CHECK-NEXT: %3 = getelementptr inbounds [13 x i8], [13 x i8]* @3, i8 0, i8 0
  ; CHECK-NEXT: call void @avr_instrumentation_end_signature(i8* %3, i16 2)

  ; CHECK-NEXT: br label %4

  ; CHECK: call void @avr_instrumentation_result_i8(i8 1)
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

  ; CHECK:        call void @avr_instrumentation_result_i32(i32 50)
  ; CHECK-NEXT:   ret i32 50
  ret i32 50
}

; CHECK-LABEL: floaty
define float @floaty(float %a) {
  ; CHECK: instrumentation_entry:
  ; CHECK-NEXT:   %0 = getelementptr inbounds [7 x i8], [7 x i8]* @6, i8 0, i8 0
  ; CHECK-NEXT:   call void @avr_instrumentation_begin_signature(i8* %0, i16 1)
  ; CHECK-NEXT:   %1 = getelementptr inbounds [2 x i8], [2 x i8]* @7, i8 0, i8 0
  ; CHECK-NEXT:   call void @avr_instrumentation_argument_f32(i8* %1, i8 0, float %a)
  ; CHECK-NEXT:   %2 = getelementptr inbounds [7 x i8], [7 x i8]* @8, i8 0, i8 0
  ; CHECK-NEXT:   call void @avr_instrumentation_end_signature(i8* %2, i16 1)

  ; CHECK-NEXT:   br label %3
  ;
  ; CHECK:        call void @avr_instrumentation_result_f32(float 1.200000e+01)
  ; CHECK-NEXT:   ret float 1.200000e+01
  ret float 12.0
}
