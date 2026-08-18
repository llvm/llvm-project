; Test each TLS size option
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -tls-size=12 < %s | FileCheck %s --check-prefix=CHECK-12
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -tls-size=12 | llvm-objdump -r - | FileCheck --check-prefix=CHECK-12-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=tiny -tls-size=24 < %s | FileCheck %s --check-prefix=CHECK-24
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=tiny -tls-size=24 | llvm-objdump -r - | FileCheck --check-prefix=CHECK-24-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=small -tls-size=32 < %s | FileCheck %s --check-prefix=CHECK-32
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=small -tls-size=32 | llvm-objdump -r - | FileCheck --check-prefix=CHECK-32-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=large -tls-size=48 < %s | FileCheck %s --check-prefix=CHECK-48
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=large -tls-size=48 | llvm-objdump -r - | FileCheck --check-prefix=CHECK-48-RELOC %s
;
; Test the maximum TLS size for each code model (fallback to a smaller size from the specified size)
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -tls-size=32 < %s | FileCheck %s --check-prefix=CHECK-32
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -tls-size=32 | llvm-objdump -r - | FileCheck --check-prefix=CHECK-32-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=tiny -tls-size=32 < %s | FileCheck %s --check-prefix=CHECK-24
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=tiny -tls-size=32 | llvm-objdump -r - | FileCheck --check-prefix=CHECK-24-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=small -tls-size=48 < %s | FileCheck %s --check-prefix=CHECK-32
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=small -tls-size=48 | llvm-objdump -r - | FileCheck --check-prefix=CHECK-32-RELOC %s
;
; Test the default TLS size for each code model
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding < %s | FileCheck --check-prefix=CHECK-24 %s
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-24-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=tiny < %s | FileCheck %s --check-prefix=CHECK-24
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=tiny | llvm-objdump -r - | FileCheck --check-prefix=CHECK-24-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=small < %s | FileCheck %s --check-prefix=CHECK-24
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=small | llvm-objdump -r - | FileCheck --check-prefix=CHECK-24-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding -code-model=large < %s | FileCheck %s --check-prefix=CHECK-24
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s -code-model=large | llvm-objdump -r - | FileCheck --check-prefix=CHECK-24-RELOC %s

@local_exec_var = thread_local(localexec) global i32 0
@local_exec_var64 = thread_local(localexec) global i64 0
@vec_local_exec_var = thread_local(localexec) global <2 x i64> zeroinitializer, align 16

define i32 @test_local_exec() {
; CHECK-LABEL: test_local_exec:
  %val = load i32, ptr @local_exec_var

; CHECK-12: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-12: ldr w0, [x[[R1]], :tprel_lo12:local_exec_var]

; CHECK-12-RELOC: R_AARCH64_TLSLE_LDST32_TPREL_LO12

; CHECK-24: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-24: add x[[R2:[0-9]+]], x[[R1]], :tprel_hi12:local_exec_var
; CHECK-24: ldr w0, [x[[R2]], :tprel_lo12_nc:local_exec_var]

; CHECK-24-RELOC: R_AARCH64_TLSLE_ADD_TPREL_HI12
; CHECK-24-RELOC: R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC

; CHECK-32: movz x[[R2:[0-9]+]], #:tprel_g1:local_exec_var
; CHECK-32: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-32: movk x[[R2]], #:tprel_g0_nc:local_exec_var
; CHECK-32: ldr w0, [x[[R1]], x[[R2]]]

; CHECK-32-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G1
; CHECK-32-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G0_NC

; CHECK-48: movz x[[R2:[0-9]+]], #:tprel_g2:local_exec_var
; CHECK-48: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-48: movk x[[R2]], #:tprel_g1_nc:local_exec_var
; CHECK-48: movk x[[R2]], #:tprel_g0_nc:local_exec_var
; CHECK-48: ldr w0, [x[[R1]], x[[R2]]]

; CHECK-48-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G2
; CHECK-48-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G1_NC
; CHECK-48-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G0_NC
  ret i32 %val
}

define void @test_local_exec_store64(i64 %val) {
; CHECK-LABEL: test_local_exec_store64:
  store i64 %val, ptr @local_exec_var64

; CHECK-12: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-12: str x0, [x[[R1]], :tprel_lo12:local_exec_var64]

; CHECK-12-RELOC: R_AARCH64_TLSLE_LDST64_TPREL_LO12

; CHECK-24: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-24: add x[[R2:[0-9]+]], x[[R1]], :tprel_hi12:local_exec_var64
; CHECK-24: str x0, [x[[R2]], :tprel_lo12_nc:local_exec_var64]

; CHECK-24-RELOC: R_AARCH64_TLSLE_ADD_TPREL_HI12 local_exec_var64
; CHECK-24-RELOC-NEXT: R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC local_exec_var64
  ret void
}

define ptr @test_local_exec_addr() {
; CHECK-LABEL: test_local_exec_addr:
  ret ptr @local_exec_var

; CHECK-12: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-12: add x0, x[[R1]], :tprel_lo12:local_exec_var
; CHECK-12: ret

; CHECK-12-RELOC: R_AARCH64_TLSLE_ADD_TPREL_LO12

; CHECK-24: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-24: add x[[R2:[0-9]+]], x[[R1]], :tprel_hi12:local_exec_var
; CHECK-24: add x0, x[[R2]], :tprel_lo12_nc:local_exec_var
; CHECK-24: ret

; CHECK-24-RELOC: R_AARCH64_TLSLE_ADD_TPREL_HI12
; CHECK-24-RELOC: R_AARCH64_TLSLE_ADD_TPREL_LO12_NC

; CHECK-32: movz x[[R2:[0-9]+]], #:tprel_g1:local_exec_var
; CHECK-32: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-32: movk x[[R2]], #:tprel_g0_nc:local_exec_var
; CHECK-32: add x0, x[[R1]], x[[R2]]
; CHECK-32: ret

; CHECK-32-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G1
; CHECK-32-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G0_NC

; CHECK-48: movz x[[R2:[0-9]+]], #:tprel_g2:local_exec_var
; CHECK-48: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-48: movk x[[R2]], #:tprel_g1_nc:local_exec_var
; CHECK-48: movk x[[R2]], #:tprel_g0_nc:local_exec_var
; CHECK-48: add x0, x[[R1]], x[[R2]]
; CHECK-48: ret

; CHECK-48-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G2
; CHECK-48-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G1_NC
; CHECK-48-RELOC: R_AARCH64_TLSLE_MOVW_TPREL_G0_NC
}

; A 128-bit access would need R_AARCH64_TLSLE_LDST128_TPREL_LO12 or its NC
; variant, which not every linker implements, so the low part stays in a
; separate add.
define <2 x i64> @test_local_exec_128bit() {
; CHECK-LABEL: test_local_exec_128bit:
  %val = load <2 x i64>, ptr @vec_local_exec_var

; CHECK-12: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-12: add x[[R2:[0-9]+]], x[[R1]], :tprel_lo12:vec_local_exec_var
; CHECK-12: ldr q0, [x[[R2]]]

; CHECK-12-RELOC: R_AARCH64_TLSLE_ADD_TPREL_LO12 vec_local_exec_var

; CHECK-24: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK-24: add x[[R2:[0-9]+]], x[[R1]], :tprel_hi12:vec_local_exec_var
; CHECK-24: add x[[R3:[0-9]+]], x[[R2]], :tprel_lo12_nc:vec_local_exec_var
; CHECK-24: ldr q0, [x[[R3]]]

; CHECK-24-RELOC: R_AARCH64_TLSLE_ADD_TPREL_HI12 vec_local_exec_var
; CHECK-24-RELOC-NEXT: R_AARCH64_TLSLE_ADD_TPREL_LO12_NC vec_local_exec_var
  ret <2 x i64> %val
}
