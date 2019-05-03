; RUN: llc -global-isel -stop-after irtranslator -mtriple aarch64-apple-ios %s -o - | FileCheck %s

; We use to incorrectly use the store size instead of the alloc size when
; creating the stack slot for allocas. This shows on aarch64 only when
; we allocated weirdly sized type. For instance, in that case, we used
; to allocate a slot of size 24-bit (19 rounded up to the next byte),
; whereas we really want to use a full 32-bit slot for this type.
; CHECK-LABEL: foo
; Check that the stack slot is 4-byte wide instead of the previously
; wrongly 3-byte sized slot.
; CHECK: stack:
; CHECK-NEXT: - { id: 0, name: stack_slot, type: default, offset: 0, size: 4, alignment: 4
define void @foo() {
  %stack_slot = alloca i19
  call void @bar(i19* %stack_slot)
  ret void
}

declare void @bar(i19* %a)
