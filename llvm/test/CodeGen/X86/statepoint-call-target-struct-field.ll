; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; The call target of a gc.statepoint may be a field of a by-value aggregate
; argument, which SelectionDAG lowers to a (possibly non-zero) result of a
; merge_values node.  LowerAsSTATEPOINT must record that exact SDValue as the
; call target; it previously rebuilt the operand with result number forced to
; 0, which silently calls the aggregate's *first* field instead of the intended
; one.  Here the two statepoints must call through %rsi (field 1) and %rdi
; (field 0) respectively.

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)

define void @statepoint_call_target_struct_field({ ptr, ptr } %fns, i1 %cond) gc "statepoint-example" {
; CHECK-LABEL: statepoint_call_target_struct_field:
; CHECK:       callq *%rsi
; CHECK:       callq *%rdi
entry:
  br i1 %cond, label %call_field1, label %call_field0

call_field1:
  %f1 = extractvalue { ptr, ptr } %fns, 1
  call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) %f1, i32 0, i32 0, i32 0, i32 0) [ "deopt"() ]
  ret void

call_field0:
  %f0 = extractvalue { ptr, ptr } %fns, 0
  call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 1, i32 0, ptr elementtype(void ()) %f0, i32 0, i32 0, i32 0, i32 0) [ "deopt"() ]
  ret void
}
