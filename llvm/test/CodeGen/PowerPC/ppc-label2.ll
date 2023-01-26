; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -ppc-asm-full-reg-names -relocation-model=pic | FileCheck --check-prefix=CHECK-PIC32 %s
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -ppc-asm-full-reg-names -relocation-model=static | FileCheck --check-prefix=CHECK-STATIC32 %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -ppc-asm-full-reg-names -relocation-model=pic | FileCheck --check-prefix=CHECK-PPC64 %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -ppc-asm-full-reg-names -relocation-model=static | FileCheck --check-prefix=CHECK-PPC64 %s


define dso_local i64 @foo() {
entry:
  br label %__here

__here:                                           ; preds = %entry
  ret i64 ptrtoint (ptr blockaddress(@foo, %__here) to i64)
}

; CHECK-PIC32:           lwz {{r[0-9]+}}, .LC0-.LTOC(r30)
; CHECK-PIC32-NOT:       li {{r[0-9]+}}, .Ltmp1-.L1$pb@l
; CHECK-PIC32-NOT:       addis 4, 30, .Ltmp1-.L1$pb@ha
; CHECK-STATIC32:        li {{r[0-9]+}}, .Ltmp0@l
; CHECK-STATIC32-NEXT:   addis {{r[0-9]+}}, {{r[0-9]+}}, .Ltmp0@ha
; CHECK-PPC64:           addis   r3, r2, .LC0@toc@ha
; CHECK-PPC64-NEXT:      ld r3, .LC0@toc@l(r3)