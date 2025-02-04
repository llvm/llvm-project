; RUN: llc %s -o - -verify-machineinstrs -mtriple=aarch64 -mattr=+lse -O0 | FileCheck %s
; RUN: llc %s -o - -verify-machineinstrs -mtriple=aarch64 -mattr=+lse -O1 | FileCheck %s

; When their destination register is WZR/ZZR, SWP operations are not regarded as
; a read for the purpose of a DMB.LD in the AArch64 memory model.
; This test ensures that the AArch64DeadRegisterDefinitions pass does not
; replace the desitnation register of SWP instructions with the zero register
; when the read value is unused.

define dso_local i32 @atomic_exchange_monotonic(ptr %ptr, ptr %ptr2, i32 %value) {
; CHECK-LABEL: atomic_exchange_monotonic:
; CHECK:       // %bb.0:
; CHECK-NEXT:    swp
; CHECK-NOT:     wzr
; CHECK-NEXT:    dmb ishld
; CHECK-NEXT:    ldr w0, [x1]
; CHECK-NEXT:    ret
    %r0 = atomicrmw xchg ptr %ptr, i32 %value monotonic
    fence acquire
    %r1 = load atomic i32, ptr %ptr2 monotonic, align 4
    ret i32 %r1
}

define dso_local i32 @atomic_exchange_acquire(ptr %ptr, ptr %ptr2, i32 %value) {
; CHECK-LABEL: atomic_exchange_acquire:
; CHECK:       // %bb.0:
; CHECK-NEXT:    swpa
; CHECK-NOT:     wzr
; CHECK-NEXT:    dmb ishld
; CHECK-NEXT:    ldr w0, [x1]
; CHECK-NEXT:    ret
    %r0 = atomicrmw xchg ptr %ptr, i32 %value acquire
    fence acquire
    %r1 = load atomic i32, ptr %ptr2 monotonic, align 4
    ret i32 %r1
}

define dso_local i32 @atomic_exchange_release(ptr %ptr, ptr %ptr2, i32 %value) {
; CHECK-LABEL: atomic_exchange_release:
; CHECK:       // %bb.0:
; CHECK-NEXT:    swpl
; CHECK-NOT:     wzr
; CHECK-NEXT:    dmb ishld
; CHECK-NEXT:    ldr w0, [x1]
; CHECK-NEXT:    ret
    %r0 = atomicrmw xchg ptr %ptr, i32 %value release
    fence acquire
    %r1 = load atomic i32, ptr %ptr2 monotonic, align 4
    ret i32 %r1
}

define dso_local i32 @atomic_exchange_acquire_release(ptr %ptr, ptr %ptr2, i32 %value) {
; CHECK-LABEL: atomic_exchange_acquire_release:
; CHECK:       // %bb.0:
; CHECK-NEXT:    swpal
; CHECK-NOT:     wzr
; CHECK-NEXT:    dmb ishld
; CHECK-NEXT:    ldr w0, [x1]
; CHECK-NEXT:    ret
    %r0 = atomicrmw xchg ptr %ptr, i32 %value acq_rel
    fence acquire
    %r1 = load atomic i32, ptr %ptr2 monotonic, align 4
    ret i32 %r1
}
