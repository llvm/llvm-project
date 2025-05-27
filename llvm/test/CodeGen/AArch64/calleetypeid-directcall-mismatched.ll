;; Tests that callee_type metadata attached to direct call sites are safely ignored.

; RUN: llc --call-graph-section -mtriple aarch64-linux-gnu < %s -stop-after=finalize-isel -o - | FileCheck %s

; Function Attrs: mustprogress noinline optnone uwtable
define i32 @_Z3fooiii(i32 %x, i32 %y, i32 %z) !type !3 {
entry:
  ;; Test that `calleeTypeIds` field is not present in `callSites`
  ; CHECK-LABEL: callSites:
  ; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [] }
  ; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [] }
  ; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [] }
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %z.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  store i32 %y, ptr %y.addr, align 4
  store i32 %z, ptr %z.addr, align 4
  %zval = load i32, ptr %z.addr, align 4
  %yval = load i32, ptr %y.addr, align 4    
  ;; This direct call has a callee_type metadata node which matches the
  ;; callee type accurately.
  %call = call i32 @_Z4fizzii(i32 %zval, i32 %yval), !callee_type !0
  %xval = load i32, ptr %x.addr, align 4
  %yval2 = load i32, ptr %y.addr, align 4
  ;; This direct call has a callee_type metadata node which points to a
  ;; mismatched callee type id.
  %call1 = call i32 @_Z4fizzii(i32 %xval, i32 %yval2), !callee_type !1
  %add = add nsw i32 %call, %call1
  %xval2 = load i32, ptr %x.addr, align 4
  %zval2 = load i32, ptr %z.addr, align 4
  ;; This direct call has a callee_type metadata node which points to a
  ;; mismatched callee type id.
  %call2 = call i32 @_Z4fizzii(i32 %xval2, i32 %zval2), !callee_type !1
  %sub = sub nsw i32 %add, %call2
  ret i32 %sub
}

declare !type !4 i32 @_Z4fizzii(i32, i32)

!0 = !{!4}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{i64 0, !"_ZTSFiiiiE.generalized"}
!4 = !{i64 0, !"_ZTSFiiiE.generalized"}
