; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-ios %s -stop-after=irtranslator -o - -global-isel | FileCheck %s


; CHECK: name: test_stack_guard

; CHECK: frameInfo:
; CHECK: stackProtector:  '%stack.0.StackGuardSlot'

; CHECK: stack:
; CHECK:  - { id: 0, name: StackGuardSlot,  type: default, offset: 0, size: 8, alignment: 8,
; CHECK-NOT: id: 1

; CHECK: [[GUARD_SLOT:%[0-9]+]]:_(p0) = G_FRAME_INDEX %stack.0.StackGuardSlot
; CHECK: [[GUARD:%[0-9]+]]:gpr64sp(p0) = LOAD_STACK_GUARD :: (dereferenceable invariant load (p0) from @__stack_chk_guard)
; CHECK: G_STORE [[GUARD]](p0), [[GUARD_SLOT]](p0) :: (volatile store (p0) into %stack.0.StackGuardSlot)
declare void @llvm.stackprotector(ptr, ptr)
define void @test_stack_guard_remat2() {
  %StackGuardSlot = alloca ptr
  call void @llvm.stackprotector(ptr undef, ptr %StackGuardSlot)
  ret void
}

@__stack_chk_guard = external global ptr
