; RUN: llc -march=mips -mattr=+o32 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-EB
; RUN: llc -march=mipsel -mattr=+o32 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-EL
; RUN: llc -march=mips -mattr=+o32,+fpxx < %s | FileCheck %s --check-prefixes=CHECK,CHECK-EB
; RUN: llc -march=mipsel -mattr=+o32,+fpxx < %s | FileCheck %s --check-prefixes=CHECK,CHECK-EL
; RUN: llc -march=mips -mattr=+o32,+fp64,+mips32r2 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-EB
; RUN: llc -march=mipsel -mattr=+o32,+fp64,+mips32r2 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-EL

@var = global double 0.0

declare void @foo(...)

define void @bar() {

; CHECK-LABEL:  bar:

; CHECK:  .cfi_def_cfa_offset 40
; CHECK:  sdc1  $f22, 32($sp)
; CHECK:  sdc1  $f20, 24($sp)
; CHECK:  sw  $ra, 20($sp)
; CHECK:  sw  $16, 16($sp)

; CHECK-EB:  .cfi_offset 55, -8
; CHECK-EB:  .cfi_offset 54, -4
; CHECK-EB:  .cfi_offset 53, -16
; CHECK-EB:  .cfi_offset 52, -12

; CHECK-EL:  .cfi_offset 54, -8
; CHECK-EL:  .cfi_offset 55, -4
; CHECK-EL:  .cfi_offset 52, -16
; CHECK-EL:  .cfi_offset 53, -12

; CHECK:  .cfi_offset 31, -20
; CHECK:  .cfi_offset 16, -24

    %val1 = load volatile double, ptr @var
    %val2 = load volatile double, ptr @var
    call void (...) @foo() nounwind
    store volatile double %val1, ptr @var
    store volatile double %val2, ptr @var
    ret void
}
