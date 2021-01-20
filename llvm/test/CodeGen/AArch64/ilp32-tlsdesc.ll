; RUN: llc -mtriple=aarch64-linux-gnu_ilp32 -relocation-model=pic %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu_ilp32 -relocation-model=pic -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s

@var = thread_local global i32 zeroinitializer

define i32 @test_thread_local() {
; CHECK-LABEL: test_thread_local:

  %val = load i32, i32* @var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc:var
; CHECK-NEXT: ldr w[[CALLEE:[0-9]+]], [x[[TLSDESC_HI]], :tlsdesc_lo12:var]
; CHECK-NEXT: add w0, w[[TLSDESC_HI]], :tlsdesc_lo12:var
; CHECK-NEXT: .tlsdesccall var
; CHECK-NEXT: blr x[[CALLEE]]

; CHECK-RELOC: R_AARCH64_P32_TLSDESC_ADR_PAGE21
; CHECK-RELOC: R_AARCH64_P32_TLSDESC_LD32_LO12
; CHECK-RELOC: R_AARCH64_P32_TLSDESC_ADD_LO12
; CHECK-RELOC: R_AARCH64_P32_TLSDESC_CALL
}
