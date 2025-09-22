;; Test if the callee_type metadata attached to indirect call sites adhere to the expected format.

; RUN: llvm-as < %s | llvm-dis | FileCheck %s
define i32 @_Z13call_indirectPFicEc(ptr %func, i8 signext %x) !type !0 {
entry:
  %func.addr = alloca ptr, align 8
  %x.addr = alloca i8, align 1
  store ptr %func, ptr %func.addr, align 8
  store i8 %x, ptr %x.addr, align 1
  %fptr = load ptr, ptr %func.addr, align 8
  %x_val = load i8, ptr %x.addr, align 1
  ; CHECK: %call = call i32 %fptr(i8 signext %x_val), !callee_type !1
  %call = call i32 %fptr(i8 signext %x_val), !callee_type !1
  ret i32 %call
}

declare !type !2 i32 @_Z3barc(i8 signext)

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
