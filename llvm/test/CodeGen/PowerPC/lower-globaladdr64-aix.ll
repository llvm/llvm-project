; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -code-model=small \
; RUN: -stop-after=machine-cp -print-before=simple-register-coalescing 2>&1 < \
; RUN: %s | FileCheck --check-prefix=SMALL %s

; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff -code-model=medium \
; RUN: -stop-after=machine-cp 2>&1 < %s | FileCheck --check-prefix=MEDIUM %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -code-model=large \
; RUN: -stop-after=machine-cp -print-before=simple-register-coalescing 2>&1 < \
; RUN: %s | FileCheck --check-prefix=LARGE %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp \
; RUN: -print-before=simple-register-coalescing 2>&1 < %s | FileCheck \
; RUN: --check-prefix=SMALL %s

@msg = common global i8* null, align 8
@ptr = common global i8* null, align 8

define void @foo() {
entry:
; SMALL: %0:g8rc_and_g8rc_nox0 = LDtoc @msg, $x2 :: (load 8 from got)
; SMALL: %1:g8rc = LD 0, %0:g8rc_and_g8rc_nox0 :: (dereferenceable load 8 from @msg)
; SMALL: %2:g8rc_and_g8rc_nox0 = LDtoc @ptr, $x2 :: (load 8 from got)
; SMALL: STD %1:g8rc, 0, %2:g8rc_and_g8rc_nox0 :: (store 8 into @ptr)

; MEDIUM: Medium code model is not supported on AIX.

; LARGE: %0:g8rc_and_g8rc_nox0 = ADDIStocHA8 $x2, @msg
; LARGE: %1:g8rc_and_g8rc_nox0 = LDtocL @msg, %0:g8rc_and_g8rc_nox0, implicit $x2 :: (load 8 from got)
; LARGE: %2:g8rc = LD 0, %1:g8rc_and_g8rc_nox0 :: (dereferenceable load 8 from @msg)
; LARGE: %3:g8rc_and_g8rc_nox0 = ADDIStocHA8 $x2, @ptr
; LARGE: %4:g8rc_and_g8rc_nox0 = LDtocL @ptr, %3:g8rc_and_g8rc_nox0, implicit $x2 :: (load 8 from got)
; LARGE: STD %2:g8rc, 0, %4:g8rc_and_g8rc_nox0 :: (store 8 into @ptr)

  %0 = load i8*, i8** @msg, align 8
  store i8* %0, i8** @ptr, align 8
  ret void
}
