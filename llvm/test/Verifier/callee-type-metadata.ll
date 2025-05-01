;; Test if the callee_type metadata attached to indirect call sites adhere to the expected format.

; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s
define i32 @_Z13call_indirectPFicEc(ptr %func, i8 signext %x) !type !0 {
entry:
  %func.addr = alloca ptr, align 8
  %x.addr = alloca i8, align 1
  store ptr %func, ptr %func.addr, align 8
  store i8 %x, ptr %x.addr, align 1
  %fptr = load ptr, ptr %func.addr, align 8
  %x_val = load i8, ptr %x.addr, align 1
  ;; No failures expected for this callee_type metdata.
  %call = call i32 %fptr(i8 signext %x_val), !callee_type !1
  ;; callee_type metdata is a type metadata instead of a list of type metadata nodes.
  ; CHECK: The callee_type metadata must be a list of type metadata nodes
  %call2 = call i32 %fptr(i8 signext %x_val), !callee_type !0
  ;; callee_type metdata is a list of non "gneralized" type metadata.  
  ; CHECK: Only generalized type metadata can be part of the callee_type metadata list
  %call3 = call i32 %fptr(i8 signext %x_val), !callee_type !4
  ;; callee_type metadata should not be part of a direct call.
  ; CHECK: !callee_type metadata should only exist on indirect function calls
  %call_direct = call i32 @_Z3barc(i8 signext %x_val), !callee_type !1
  ;; callee_type metadata should not be part of a direct call.
  ; CHECK: !callee_type metadata should only exist on indirect function calls
  %call_direct_wrong_signature = call i32 @_Z3barc(i8 signext %x_val), !callee_type !4
  ret i32 %call
}

declare !type !2 i32 @_Z3barc(i8 signext)

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{i64 0, !"_ZTSFicE"}
!4 = !{!3}
!8 = !{i64 0, !"_ZTSFicE.generalized"}
