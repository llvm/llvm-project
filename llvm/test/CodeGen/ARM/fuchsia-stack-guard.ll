; RUN: llc -mtriple=thumbv8a-unknown-fuchsia -mattr=+read-tp-tpidruro -O2 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-TPIDRURO
; RUN: llc -mtriple=armv8a-unknown-fuchsia -mattr=+read-tp-tpidruro -O2 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-TPIDRURO
; RUN: llc -mtriple=thumbv8a-unknown-fuchsia -mattr=+read-tp-tpidruro -O0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-TPIDRURO
; RUN: llc -mtriple=armv8a-unknown-fuchsia -mattr=+read-tp-tpidruro -O0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-TPIDRURO
; RUN: llc -mtriple=thumbv8a-unknown-fuchsia -O2 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-SOFT

declare void @sink(ptr)

define void @test_ssp() sspreq {
; CHECK-LABEL: test_ssp:
; CHECK-NOT: __stack_chk_guard
; CHECK-TPIDRURO: mrc p15, #0, [[TP:r[0-9]+]], c13, c0, #3
; CHECK-TPIDRURO: ldr{{(\.w)?}} {{r[0-9]+}}, [[[TP]], #-16]
; CHECK-SOFT: bl __aeabi_read_tp
; CHECK-SOFT: ldr{{(\.w)?}} {{r[0-9]+}}, [r0, #-16]
; CHECK: bl sink
; CHECK: ldr{{(\.w)?}} {{r[0-9]+}}, [{{r[0-9]+}}, #-16]
; CHECK: bl __stack_chk_fail
; CHECK-NOT: __stack_chk_guard
  %buf = alloca [16 x i8], align 4
  call void @sink(ptr nonnull %buf)
  ret void
}
