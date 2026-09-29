; RUN: llc -mtriple=armv4-none--eabi < %s | FileCheck %s --check-prefix=CHECK-LOLOMOV
; RUN: llc -mtriple=armv4t-none--eabi < %s | FileCheck %s --check-prefix=CHECK-LOLOMOV
; RUN: llc -mtriple=armv5-none--eabi < %s | FileCheck %s --check-prefix=CHECK-LOLOMOV
; RUN: llc -mtriple=armv6-none--eabi < %s | FileCheck %s --check-prefix=CHECK-LOLOMOV
; RUN: llc -mtriple=armv7-none--eabi < %s | FileCheck %s --check-prefix=CHECK-LOLOMOV
; RUN: llc -mtriple=thumbv6-none--eabi < %s | FileCheck %s --check-prefix=CHECK-LOLOMOV
; RUN: llc -mtriple=thumbv7-none--eabi < %s | FileCheck %s --check-prefix=CHECK-LOLOMOV
; CHECK-LOLOMOV-LABEL:  foo
; CHECK-LOLOMOV:        mov [[TMP:r[0-7]]], [[SRC1:r[01]]]
; CHECK-LOLOMOV-NEXT:   mov [[SRC1]], [[SRC2:r[01]]]
; CHECK-LOLOMOV-NEXT:   mov [[SRC2]], [[TMP]]
; CHECK-LOLOMOV-LABEL:  bar
; CHECK-LOLOMOV-LABEL:  fnend
; 
; 'MOV lo, lo' in Thumb mode produces undefined results on pre-v6 hardware
; RUN: llc -mtriple=thumbv4t-none--eabi < %s | FileCheck %s --check-prefix=CHECK-NOLOLOMOV
; RUN: llc -mtriple=thumbv5-none--eabi < %s | FileCheck %s --check-prefix=CHECK-NOLOLOMOV
; CHECK-NOLOLOMOV-LABEL: foo
; CHECK-NOLOLOMOV:       movs [[TMP:r[0-7]]], [[SRC1:r[01]]]
; CHECK-NOLOLOMOV-NEXT:  movs [[SRC1]], [[SRC2:r[01]]]
; CHECK-NOLOLOMOV-NEXT:  movs [[SRC2]], [[TMP]]
; CHECK-NOLOLOMOV-LABEL: bar
; CHECK-NOLOLOMOV-LABEL: fnend

declare void @bar(i32, i32)

define void @foo(i32 %a, i32 %b) {
entry:
  call void @bar(i32 %b, i32 %a);
  ret void
}

