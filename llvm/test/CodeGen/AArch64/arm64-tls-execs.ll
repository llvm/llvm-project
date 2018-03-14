; RUN: llc -mtriple=arm64-none-linux-gnu -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; RUN: llc -mtriple=arm64-none-linux-gnu -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s

@initial_exec_var = external thread_local(initialexec) global i32

define i32 @test_initial_exec() {
; CHECK-LABEL: test_initial_exec:
  %val = load i32, i32* @initial_exec_var

; CHECK: adrp x[[GOTADDR:[0-9]+]], :gottprel:initial_exec_var
; CHECK: ldr x[[TP_OFFSET:[0-9]+]], [x[[GOTADDR]], :gottprel_lo12:initial_exec_var]
; CHECK: mrs x[[TP:[0-9]+]], TPIDR_EL0
; CHECK: ldr w0, [x[[TP]], x[[TP_OFFSET]]]

; CHECK-RELOC: R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
; CHECK-RELOC: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC

  ret i32 %val
}

define i32* @test_initial_exec_addr() {
; CHECK-LABEL: test_initial_exec_addr:
  ret i32* @initial_exec_var

; CHECK: adrp x[[GOTADDR:[0-9]+]], :gottprel:initial_exec_var
; CHECK: ldr [[TP_OFFSET:x[0-9]+]], [x[[GOTADDR]], :gottprel_lo12:initial_exec_var]
; CHECK: mrs [[TP:x[0-9]+]], TPIDR_EL0
; CHECK: add x0, [[TP]], [[TP_OFFSET]]

; CHECK-RELOC: R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
; CHECK-RELOC: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC

}

@local_exec_var = thread_local(localexec) global i32 0

define i32 @test_local_exec() {
; CHECK-LABEL: test_local_exec:
  %val = load i32, i32* @local_exec_var

; CHECK: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK: add x[[R2:[0-9]+]], x[[R1]], :tprel_hi12:local_exec_var
; CHECK: add x[[R3:[0-9]+]], x[[R2]], :tprel_lo12_nc:local_exec_var
; CHECK: ldr w0, [x[[R3]]]

; CHECK-RELOC: R_AARCH64_TLSLE_ADD_TPREL_HI12
; CHECK-RELOC: R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
  ret i32 %val
}

define i32* @test_local_exec_addr() {
; CHECK-LABEL: test_local_exec_addr:
  ret i32* @local_exec_var

; CHECK: mrs x[[R1:[0-9]+]], TPIDR_EL0
; CHECK: add x[[R2:[0-9]+]], x[[R1]], :tprel_hi12:local_exec_var
; CHECK: add x0, x[[R2]], :tprel_lo12_nc:local_exec_var
; CHECK: ret

; CHECK-RELOC: R_AARCH64_TLSLE_ADD_TPREL_HI12
; CHECK-RELOC: R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
}
