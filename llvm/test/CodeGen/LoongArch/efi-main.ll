; RUN: llc --mtriple=loongarch64-unknown-uefi < %s | FileCheck %s

define i64 @efi_main(ptr noundef %handle, ptr noundef %system_table) {
entry:
  %system_table.addr = alloca ptr, align 8
  %handle.addr = alloca ptr, align 8
  store ptr %system_table, ptr %system_table.addr, align 8
  store ptr %handle, ptr %handle.addr, align 8
  ret i64 0
}

; UEFI:        .section .text.efi,"ax",@progbits
; CHECK-LABEL: efi_main:
; CHECK:       addi.d  $sp, $sp, -16
; CHECK:       st.d    $a1, $sp, 8
; CHECK:       st.d    $a0, $sp, 0
; CHECK:       move    $a0, $zero
; CHECK:       addi.d  $sp, $sp, 16
; CHECK:       ret
