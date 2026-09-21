; RUN: llc -mtriple s390x-zos < %s | FileCheck %s

; Checks that the 2 C_WSA64 classes for the DWARF EH code are not merged together.

define { ptr, i32 } @fn1() personality ptr @__zos_cxx_personality_v2 {
start:
  %_0 = invoke double null(double 0.0, double 0.0, double 0.0)
          to label %bb1 unwind label %terminate
terminate:
  %0 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  ret { ptr, i32 } %0
bb1:
  ret { ptr, i32 } zeroinitializer
}

define { ptr, i32 } @fn2() personality ptr @__zos_cxx_personality_v2 {
start:
  %_0 = invoke double null(double 0.0, double 0.0, double 0.0)
          to label %bb1 unwind label %terminate
terminate:
  %0 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  ret { ptr, i32 } %0
bb1:
  ret { ptr, i32 } zeroinitializer
}

declare i32 @__zos_cxx_personality_v2(...)

; CHECK:      C_WSA64 CATTR ALIGN(2),FILL(0),NOTEXECUTABLE,RMODE(64),PART(.gcc_excepti
; CHECK-NEXT:                ion_table.fn1)
; CHECK-NEXT: .gcc_exception_table.fn1 XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SEC
; CHECK-NEXT:                CTION)

; CHECK:      C_WSA64 CATTR ALIGN(2),FILL(0),NOTEXECUTABLE,RMODE(64),PART(.gcc_excepti
; CHECK-NEXT:                ion_table.fn2)
; CHECK-NEXT: .gcc_exception_table.fn2 XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SEC
; CHECK-NEXT:                CTION)
