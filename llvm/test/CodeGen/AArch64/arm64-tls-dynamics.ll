; RUN: llc -mtriple=arm64-none-linux-gnu -relocation-model=pic -aarch64-elf-ldtls-generation=1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=arm64-none-linux-gnu -relocation-model=pic -aarch64-elf-ldtls-generation=1 -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s
; RUN: llc -mtriple=arm64-none-linux-gnu -relocation-model=pic -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK-NOLD %s
; RUN: llc -mtriple=arm64-none-linux-gnu -relocation-model=pic -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-NOLD-RELOC %s
; FIXME: We currently produce "small" code for the tiny model
; RUN: llc -mtriple=arm64-none-linux-gnu -relocation-model=pic -aarch64-elf-ldtls-generation=1 -code-model=tiny -verify-machineinstrs < %s | FileCheck %s
; FIXME: We currently error for the large code model
; RUN: not llc -mtriple=arm64-none-linux-gnu -relocation-model=pic -aarch64-elf-ldtls-generation=1 -code-model=large -verify-machineinstrs < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LARGE

; CHECK-LARGE: ELF TLS only supported in small memory model

@general_dynamic_var = external thread_local global i32

define i32 @test_generaldynamic() {
; CHECK-LABEL: test_generaldynamic:

  %val = load i32, i32* @general_dynamic_var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:general_dynamic_var
; CHECK-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:general_dynamic_var]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:general_dynamic_var
; CHECK-NEXT: .tlsdesccall general_dynamic_var
; CHECK-NEXT: blr [[CALLEE]]

; CHECK-NOLD: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:general_dynamic_var
; CHECK-NOLD-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:general_dynamic_var]
; CHECK-NOLD-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:general_dynamic_var
; CHECK-NOLD-NEXT: .tlsdesccall general_dynamic_var
; CHECK-NOLD-NEXT: blr [[CALLEE]]


; CHECK: mrs x[[TP:[0-9]+]], TPIDR_EL0
; CHECK: ldr w0, [x[[TP]], x0]
; CHECK-NOLD: mrs x[[TP:[0-9]+]], TPIDR_EL0
; CHECK-NOLD: ldr w0, [x[[TP]], x0]

; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL

; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_CALL

}

define i32* @test_generaldynamic_addr() {
; CHECK-LABEL: test_generaldynamic_addr:

  ret i32* @general_dynamic_var

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:general_dynamic_var
; CHECK-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:general_dynamic_var]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:general_dynamic_var
; CHECK-NEXT: .tlsdesccall general_dynamic_var
; CHECK-NEXT: blr [[CALLEE]]

; CHECK: mrs [[TP:x[0-9]+]], TPIDR_EL0
; CHECK: add x0, [[TP]], x0

; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL

; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_CALL

}

@local_dynamic_var = external thread_local(localdynamic) global i32

define i32 @test_localdynamic() {
; CHECK-LABEL: test_localdynamic:

  %val = load i32, i32* @local_dynamic_var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:_TLS_MODULE_BASE_
; CHECK-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:_TLS_MODULE_BASE_]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:_TLS_MODULE_BASE_
; CHECK-NEXT: .tlsdesccall _TLS_MODULE_BASE_
; CHECK-NEXT: blr [[CALLEE]]
; CHECK-NEXT: add x[[TPOFF:[0-9]+]], x0, :dtprel_hi12:local_dynamic_var
; CHECK-NEXT: add x[[TPOFF]], x[[TPOFF]], :dtprel_lo12_nc:local_dynamic_var
; CHECK: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK: ldr w0, [x[[TPIDR]], x[[TPOFF]]]

; CHECK-NOLD: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:local_dynamic_var
; CHECK-NOLD-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:local_dynamic_var]
; CHECK-NOLD-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:local_dynamic_var
; CHECK-NOLD-NEXT: .tlsdesccall local_dynamic_var
; CHECK-NOLD-NEXT: blr [[CALLEE]]
; CHECK-NOLD: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK-NOLD: ldr w0, [x[[TPIDR]], x0]


; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL
; CHECK-RELOC: R_AARCH64_TLSLD_ADD_DTPREL_HI12
; CHECK-RELOC: R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC

; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_CALL

}

define i32* @test_localdynamic_addr() {
; CHECK-LABEL: test_localdynamic_addr:

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:_TLS_MODULE_BASE_
; CHECK-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:_TLS_MODULE_BASE_]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:_TLS_MODULE_BASE_
; CHECK-NEXT: .tlsdesccall _TLS_MODULE_BASE_
; CHECK-NEXT: blr [[CALLEE]]
; CHECK-NEXT: add x[[TPOFF:[0-9]+]], x0, :dtprel_hi12:local_dynamic_var
; CHECK-NEXT: add x[[TPOFF]], x[[TPOFF]], :dtprel_lo12_nc:local_dynamic_var
; CHECK: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK: add x0, x[[TPIDR]], x[[TPOFF]]

; CHECK-NOLD: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:local_dynamic_var
; CHECK-NOLD-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:local_dynamic_var]
; CHECK-NOLD-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:local_dynamic_var
; CHECK-NOLD-NEXT: .tlsdesccall local_dynamic_var
; CHECK-NOLD-NEXT: blr [[CALLEE]]
; CHECK-NOLD: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK-NOLD: add x0, x[[TPIDR]], x0
  ret i32* @local_dynamic_var

; CHECK-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-RELOC: R_AARCH64_TLSDESC_CALL
; CHECK-RELOC: R_AARCH64_TLSLD_ADD_DTPREL_HI12
; CHECK-RELOC: R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC

; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADR_PAGE21
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_LD64_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_ADD_LO12
; CHECK-NOLD-RELOC: R_AARCH64_TLSDESC_CALL
}

; The entire point of the local-dynamic access model is to have a single call to
; the expensive resolver. Make sure we achieve that goal.

@local_dynamic_var2 = external thread_local(localdynamic) global i32

define i32 @test_localdynamic_deduplicate() {
; CHECK-LABEL: test_localdynamic_deduplicate:

  %val = load i32, i32* @local_dynamic_var
  %val2 = load i32, i32* @local_dynamic_var2

  %sum = add i32 %val, %val2
  ret i32 %sum

; CHECK: adrp x[[DTPREL_HI:[0-9]+]], :tlsdesc:_TLS_MODULE_BASE_
; CHECK-NEXT: ldr [[CALLEE:x[0-9]+]], [x[[DTPREL_HI]], :tlsdesc_lo12:_TLS_MODULE_BASE_]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_lo12:_TLS_MODULE_BASE
; CHECK-NEXT: .tlsdesccall _TLS_MODULE_BASE_
; CHECK-NEXT: blr [[CALLEE]]

; CHECK-NOT: _TLS_MODULE_BASE_

; CHECK: ret
}
