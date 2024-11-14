; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
; RUN: llc --filetype=obj %s -o - | llvm-readelf -s --unwind - | FileCheck %s --check-prefix=UNWIND
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.main-none-none-eabi"

; int g(int);
;
; #if __ARM_FEATURE_CMSE == 3
; #define ENTRY __attribute__((cmse_nonsecure_entry))
; #else
; #define ENTRY
; #endif
;
; ENTRY int f(int x) {
;     return 1 + g(x - 1);
; }

define hidden i32 @f0(i32 %x) local_unnamed_addr "sign-return-address"="non-leaf" {
entry:
  %sub = add nsw i32 %x, -1
  %call = tail call i32 @g(i32 %sub)
  %add = add nsw i32 %call, 1
  ret i32 %add
}

; CHECK-LABEL: f0:
; CHECK:       pac     r12, lr, sp
; CHECK-NEXT:  .save   {r7, lr}
; CHECK-NEXT:  push    {r7, lr}
; CHECK-NEXT: .cfi_def_cfa_offset 8
; CHECK-NEXT: .cfi_offset lr, -4
; CHECK-NEXT: .cfi_offset r7, -8
; CHECK-NEXT: .save   {ra_auth_code}
; CHECK-NEXT:  str     r12, [sp, #-4]!
; CHECK-NEXT: .cfi_def_cfa_offset 12
; CHECK-NEXT: .cfi_offset ra_auth_code, -12
; CHECK-NEXT: .pad    #4
; CHECK-NEXT:  sub     sp, #4
; ...
; CHECK:       add     sp, #4
; CHECK-NEXT:  ldr     r12, [sp], #4
; CHECK-NEXT:  pop.w   {r7, lr}
; CHECK-NEXT:  aut     r12, lr, sp
; CHECK-NEXT:  bx      lr

define hidden i32 @f1(i32 %x) local_unnamed_addr #0 {
entry:
  %sub = add nsw i32 %x, -1
  %call = tail call i32 @g(i32 %sub)
  %add = add nsw i32 %call, 1
  ret i32 %add
}

; CHECK-LABEL: f1:
; CHECK:       pac     r12, lr, sp
; CHECK-NEXT:  vstr    fpcxtns, [sp, #-4]!
; CHECK-NEXT:  .cfi_def_cfa_offset 4
; CHECK-NEXT:  .save    {r7, lr}
; CHECK-NEXT:  push    {r7, lr}
; CHECK:       vldr    fpcxtns, [sp], #4
; CHECK:       aut     r12, lr, sp

define hidden i32 @f2(i32 %x) local_unnamed_addr #1 {
entry:
  %sub = add nsw i32 %x, -1
  %call = tail call i32 @g(i32 %sub)
  %add = add nsw i32 %call, 1
  ret i32 %add
}
; CHECK-LABEL: f2:
; CHECK:       pac    r12, lr, sp
; CHECK-NEXT:  .save  {r7, lr}
; CHECK-NEXT:  push   {r7, lr}
; CHECK-NEXT: .cfi_def_cfa_offset 8
; CHECK-NEXT: .cfi_offset lr, -4
; CHECK-NEXT: .cfi_offset r7, -8
; CHECK-NEXT:  .save  {ra_auth_code}
; CHECK-NEXT:  str    r12, [sp, #-4]!
; CHECK-NEXT: .cfi_def_cfa_offset 12
; CHECK-NEXT: .cfi_offset ra_auth_code, -12
; CHECK-NEXT:  .pad   #4
; CHECK-NEXT:  sub    sp, #4
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; ...
; CHECK:       add    sp, #4
; CHECK-NEXT:  ldr    r12, [sp], #4
; CHECK-NEXT:  pop.w  {r7, lr}
; CHECK-NEXT:  aut    r12, lr, sp
; CHECK-NEXT:  mrs    r12, control
; ...
; CHECK:       bxns    lr

declare dso_local i32 @g(i32) local_unnamed_addr

attributes #0 = { "sign-return-address"="non-leaf" "cmse_nonsecure_entry" "target-features"="+8msecext,+armv8.1-m.main"}
attributes #1 = { "sign-return-address"="non-leaf" "cmse_nonsecure_entry" "target-features"="+8msecext,+armv8-m.main,+fp-armv8d16"}

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}

; UNWIND-LABEL: FunctionAddress: 0x0
; UNWIND:       0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x08 ; pop {r7, lr}
; UNWIND-NEXT:  0xB0      ; finish
; UNWIND-NEXT:  0xB0      ; finish

; UNWIND-LABEL: FunctionAddress: 0x24
; UNWIND:       0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x08 ; pop {r7, lr}

; UNWIND-LABEL: FunctionAddress: 0x54
; UNWIND:       0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x08 ; pop {r7, lr}
; UNWIND-NEXT:  0xB0      ; finish
; UNWIND-NEXT:  0xB0      ; finish

; UNWIND-LABEL: 00000001 {{.*}} f0
; UNWIND-LABEL: 00000025 {{.*}} f1
; UNWIND-LABEL: 00000055 {{.*}} f2
