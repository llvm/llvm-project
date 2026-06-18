;; Test if the callee_type metadata attached to indirect call sites adhere to the expected format.

; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s
define i32 @_Z13call_indirectPFicEc(ptr %func, i8 signext %x) !type !0 {
entry:
  %func.addr = alloca ptr, align 8
  %x.addr = alloca i8, align 1
  store ptr %func, ptr %func.addr, align 8
  store i8 %x, ptr %x.addr, align 1
  %fptr = load ptr, ptr %func.addr, align 8
  %x_val = load i8, ptr %x.addr, align 1  
  ; CHECK: The callee_type metadata must be a list of type metadata nodes
  %call = call i32 %fptr(i8 signext %x_val), !callee_type !0
  ; CHECK: Well-formed generalized type metadata must contain exactly two operands
  %call1 = call i32 %fptr(i8 signext %x_val), !callee_type !2
  ; CHECK: The first operand of type metadata for functions must be zero
  %call2 = call i32 %fptr(i8 signext %x_val), !callee_type !4
  ; CHECK: The first operand of type metadata for functions must be zero
  %call3 = call i32 %fptr(i8 signext %x_val), !callee_type !6
  ; CHECK: Only generalized type metadata can be part of the callee_type metadata list
  %call4 = call i32 %fptr(i8 signext %x_val), !callee_type !8
  ret i32 %call
}

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!"_ZTSFicE"}
!2 = !{!2}
!3 = !{i64 1, !"_ZTSFicE"}
!4 = !{!3}
!5 = !{!"expected_int", !"_ZTSFicE"}
!6 = !{!5}
!7 = !{i64 0, !"_ZTSFicE"}
!8 = !{!7}
