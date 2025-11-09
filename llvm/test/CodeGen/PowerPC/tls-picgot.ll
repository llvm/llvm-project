; RUN: llc -verify-machineinstrs -relocation-model=pic < %s | FileCheck %s

target triple = "powerpc-unknown-linux-gnu"

; Test that LR is preserved when PPC32PICGOT clobbers it with a local "bl".

@TLS = external thread_local global i8

; CHECK-LABEL: tls_addr:
; CHECK:        mflr [[SAVED_REG:[0-9]+]]

; CHECK:        bl [[JUMP:\.L[[:alnum:]_]+]]
; CHECK-NEXT:   [[OFFSET:\.L[[:alnum:]_]+]]:
; CHECK-NEXT:   .long _GLOBAL_OFFSET_TABLE_-[[OFFSET]]
; CHECK-NEXT:   [[JUMP]]
; CHECK-NEXT:   mflr {{[0-9]+}}

; CHECK:        mtlr [[SAVED_REG]]
; CHECK-NEXT:   blr

define ptr @tls_addr() unnamed_addr {
  %1 = call ptr @llvm.threadlocal.address.p0(ptr @TLS)
  ret ptr %1
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
