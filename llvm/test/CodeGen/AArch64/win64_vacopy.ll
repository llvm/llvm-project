; RUN: llc -mtriple=aarch64-pc-windows-msvc < %s | FileCheck %s --check-prefix=WIN64
; RUN: llc -mtriple=aarch64-unknown-uefi < %s | FileCheck %s --check-prefix=WIN64
; RUN: llc -mtriple=aarch64-pc-windows-msvc -O0 -global-isel < %s | FileCheck %s --check-prefix=WIN64
; RUN: llc -mtriple=aarch64-unknown-uefi -O0 -global-isel < %s | FileCheck %s --check-prefix=WIN64
; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s --check-prefix=AAPCS

; va_list is a plain pointer on Windows and UEFI, so va_copy copies a single
; pointer rather than the 32 byte AAPCS structure.
define void @copy(ptr %dst, ptr %src) {
; WIN64-LABEL: copy:
; WIN64:       ldr x8, [x1]
; WIN64-NEXT:  str x8, [x0]
; WIN64-NEXT:  ret
;
; AAPCS-LABEL: copy:
; AAPCS:       ldp q1, q0, [x1]
; AAPCS-NEXT:  stp q1, q0, [x0]
; AAPCS-NEXT:  ret
  call void @llvm.va_copy(ptr %dst, ptr %src)
  ret void
}

declare void @llvm.va_copy(ptr, ptr)
