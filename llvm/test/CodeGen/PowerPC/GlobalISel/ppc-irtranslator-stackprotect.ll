; RUN: llc -verify-machineinstrs -mtriple=ppc64le-unknown-linux %s -global-isel -stop-after=irtranslator -o - | FileCheck %s --check-prefix=LINUX
; RUN: llc -verify-machineinstrs -mtriple=ppc64le-unknown-openbsd %s -global-isel -stop-after=irtranslator -o - | FileCheck %s --check-prefix=OPENBSD


; The stack guard on Linux
@__stack_chk_guard = external global ptr

; The stack guard on OpenBSD
@__guard_local = external hidden global ptr

declare void @llvm.stackprotector(ptr, ptr)

; LINUX-LABEL: name: test_stack_guard_linux

; LINUX: frameInfo:
; LINUX: stackProtector:  '%stack.0.StackGuardSlot'

; LINUX: stack:
; LINUX:  - { id: 0, name: StackGuardSlot,  type: default, offset: 0, size: 8, alignment: 8,
; LINUX-NOT: id: 1

; LINUX: [[GUARD_SLOT:%[0-9]+]]:_(p0) = G_FRAME_INDEX %stack.0.StackGuardSlot
; LINUX: [[GUARD:%[0-9]+]]:g8rc(p0) = LOAD_STACK_GUARD :: (dereferenceable invariant load (p0) from @__stack_chk_guard)
; LINUX: G_STORE [[GUARD]](p0), [[GUARD_SLOT]](p0) :: (volatile store (p0) into %stack.0.StackGuardSlot)
define void @test_stack_guard_linux() {
  %StackGuardSlot = alloca ptr
  call void @llvm.stackprotector(ptr undef, ptr %StackGuardSlot)
  ret void
}

; OPENBSD-LABEL: name: test_stack_guard_openbsd

; OPENBSD: frameInfo:
; OPENBSD: stackProtector:  '%stack.0.StackGuardSlot'

; OPENBSD: stack:
; OPENBSD:  - { id: 0, name: StackGuardSlot,  type: default, offset: 0, size: 8, alignment: 8,
; OPENBSD-NOT: id: 1

; OPENBSD: [[GUARD_LOCAL:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @__guard_local
; OPENBSD: [[GUARD_SLOT:%[0-9]+]]:_(p0) = G_FRAME_INDEX %stack.0.StackGuardSlot
; OPENBSD: [[GUARD:%[0-9]+]]:_(p0) = G_LOAD [[GUARD_LOCAL]](p0) :: (dereferenceable load (p0) from @__guard_local)
; OPENBSD: G_STORE [[GUARD]](p0), [[GUARD_SLOT]](p0) :: (volatile store (p0) into %stack.0.StackGuardSlot)
define void @test_stack_guard_openbsd() {
  %StackGuardSlot = alloca ptr
  %StackGuard = load ptr, ptr @__guard_local
  call void @llvm.stackprotector(ptr %StackGuard, ptr %StackGuardSlot)
  ret void
}
