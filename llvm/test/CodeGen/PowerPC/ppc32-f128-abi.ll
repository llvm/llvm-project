; RUN: llc -verify-machineinstrs -mattr=+vsx < %s | FileCheck %s
target triple = "powerpc-unknown-linux-gnu"

; CHECK-LABEL: return_second:
; CHECK: vmr 2, 3
; CHECK-NEXT: blr
define fp128 @return_second(fp128, fp128) {
    ret fp128 %1
}
