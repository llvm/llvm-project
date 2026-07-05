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
  ; CHECK: The callee_type metadata must be a list of callgraph metadata nodes
  %call = call i32 %fptr(i8 signext %x_val), !callee_type !0
  ; CHECK: The operand of callgraph metadata for functions must be an MDString
  %call1 = call i32 %fptr(i8 signext %x_val), !callee_type !2
  ; CHECK: Well-formed generalized callgraph metadata must contain exactly one operand
  %call2 = call i32 %fptr(i8 signext %x_val), !callee_type !4
  ; CHECK: Only generalized callgraph metadata can be part of the callee_type metadata list
  %call3 = call i32 %fptr(i8 signext %x_val), !callee_type !6
  ret i32 %call
}

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!"_ZTSFicE"}
!2 = !{!2}
!3 = !{i64 1, !"_ZTSFicE"}
!4 = !{!3}
!5 = !{!"_ZTSFicE"}
!6 = !{!5}
