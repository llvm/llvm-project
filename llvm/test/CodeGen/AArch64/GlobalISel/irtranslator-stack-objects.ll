; RUN: llc -global-isel -mtriple=aarch64-unknown-unknown -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -global-isel -mtriple=aarch64-unknown-unknown -stop-after=irtranslator -verify-machineinstrs -opaque-pointers %s -o - | FileCheck %s

; The byval object should not be immutable, but the non-byval stack
; passed argument should be.

; CHECK-LABEL: name: stack_passed_i64
; CHECK: fixedStack:
; CHECK:  - { id: 0, type: default, offset: 8, size: 8, alignment: 8, stack-id: default,
; CHECK-NEXT:      isImmutable: false, isAliased: false,
; CHECK:  - { id: 1, type: default, offset: 0, size: 8, alignment: 16, stack-id: default,
; CHECK-NEXT: isImmutable: true, isAliased: false,
define void @stack_passed_i64(i64 %arg, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %arg5, i64 %arg6,
                              i64 %arg7, i64 %arg8, ptr byval(i64) %arg9) {
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.1
  ; CHECK:   [[LOAD:%[0-9]+]]:_(s64) = G_LOAD [[FRAME_INDEX]](p0) :: (invariant load (s64)  from %fixed-stack.1, align 16)
  ; CHECK:   [[FRAME_INDEX1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.0
  ; CHECK:   [[COPY8:%[0-9]+]]:_(p0) = COPY [[FRAME_INDEX1]](p0)
  ; CHECK:   [[LOAD1:%[0-9]+]]:_(s64) = G_LOAD [[COPY8]](p0) :: (dereferenceable load (s64)  from %ir.arg9)
  ; CHECK:   [[ADD:%[0-9]+]]:_(s64) = G_ADD [[LOAD1]], [[LOAD]]
  ; CHECK:   G_STORE [[ADD]](s64), [[COPY8]](p0) :: (volatile store (s64) into %ir.arg9)
  ; CHECK:   RET_ReallyLR
  %load = load i64, ptr %arg9
  %add = add i64 %load, %arg8
  store volatile i64 %add, ptr %arg9
  ret void
}
