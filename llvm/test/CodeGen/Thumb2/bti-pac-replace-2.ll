; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
; RUN: llc --filetype=obj %s -o - | llvm-readelf -u - | FileCheck %s --check-prefix=UNWIND
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-unknown-eabi"

define hidden i32 @_Z1fi(i32 %x) {
entry:
  %add = add nsw i32 %x, 1
  %call = tail call i32 @_Z1gi(i32 %add)
  %sub = add nsw i32 %call, -1
  ret i32 %sub
}

declare dso_local i32 @_Z1gi(i32)

!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 1, !"branch-target-enforcement", i32 1}
!1 = !{i32 1, !"sign-return-address", i32 1}
!2 = !{i32 1, !"sign-return-address-all", i32 0}

; Check the function starts with `pacbti` and correct unwind info is emitted
; CHECK-LABEL: _Z1fi:
; ...
; CHECK:       pacbti   r12, lr, sp
; CHECK-NEXT:  .save    {r7, lr}
; CHECK-NEXT:  push     {r7, lr}
; CHECK-NEXT:  .cfi_def_cfa_offset 8
; CHECK-NEXT:  .cfi_offset lr, -4
; CHECK-NEXT:  .cfi_offset r7, -8
; CHECK-NEXT:  .save   {ra_auth_code}
; CHECK-NEXT:  str     r12, [sp, #-4]!
; CHECK-NEXT:  .cfi_def_cfa_offset 12
; CHECK-NEXT:  .cfi_offset ra_auth_code, -12
; CHECK-NEXT:  .pad    #4
; CHECK-NEXT:  sub     sp, #4
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; ...

; UNWIND-LABEL: Opcodes [
; UNWIND-NEXT:  0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x08 ; pop {r7, lr}
; UNWIND-NEXT:  0xB0      ; finish
