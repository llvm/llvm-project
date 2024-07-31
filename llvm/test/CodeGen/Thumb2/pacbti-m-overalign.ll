; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
; RUN: llc --filetype=obj %s -o - | llvm-readelf --unwind - | FileCheck %s --check-prefix=UNWIND
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; int g(int, int *);
;
; int f() {
;     __attribute__((aligned(32))) int a[4];
;    g(4, a);
;   int s = 0;
;   for (int i = 0; i < 4; ++i)
;     s +=  a[i];
;   return s;
; }

define hidden i32 @_Z1fv() local_unnamed_addr "sign-return-address"="non-leaf" {
entry:
  %a = alloca [4 x i32], align 32
  %call = call i32 @_Z1giPi(i32 4, ptr nonnull %a)
  %0 = load i32, ptr %a, align 32
  %arrayidx.1 = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 1
  %1 = load i32, ptr %arrayidx.1, align 4
  %add.1 = add nsw i32 %1, %0
  %arrayidx.2 = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 2
  %2 = load i32, ptr %arrayidx.2, align 8
  %add.2 = add nsw i32 %2, %add.1
  %arrayidx.3 = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 3
  %3 = load i32, ptr %arrayidx.3, align 4
  %add.3 = add nsw i32 %3, %add.2
  ret i32 %add.3
}

; CHECK-LABEL: _Z1fv:
; CHECK:      pac     r12, lr, sp
; CHECK:      .save   {r4, r6, r7, lr}
; CHECK-NEXT: push    {r4, r6, r7, lr}
; CHECK-NEXT: .cfi_def_cfa_offset 16
; CHECK-NEXT: .cfi_offset lr, -4
; CHECK-NEXT: .cfi_offset r7, -8
; CHECK-NEXT: .cfi_offset r6, -12
; CHECK-NEXT: .cfi_offset r4, -16
; CHECK-NEXT: .setfp  r7, sp, #8
; CHECK-NEXT: add     r7, sp, #8
; CHECK-NEXT: .cfi_def_cfa r7, 8
; CHECK-NEXT: .save   {ra_auth_code}
; CHECK-NEXT: str     r12, [sp, #-4]!
; CHECK-NEXT: .cfi_offset ra_auth_code, -20
; CHECK-NEXT: .pad    #44
; CHECK-NEXT: sub     sp, #44
; CHECK:      ldr     r12, [sp], #4
; CHECK-NEXT: pop.w   {r4, r6, r7, lr}
; CHECK-NEXT: aut     r12, lr, sp
; CHECK-NEXT: bx      lr


declare dso_local i32 @_Z1giPi(i32, ptr) local_unnamed_addr

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}

; UNWIND-LABEL:        FunctionAddress: 0x0
; UNWIND:          0x97      ; vsp = r7
; UNWIND:          0x42      ; vsp = vsp - 12
; UNWIND:          0xB4      ; pop ra_auth_code
; UNWIND:          0x84 0x0D ; pop {r4, r6, r7, lr}
