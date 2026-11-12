; RUN: llc -mtriple powerpc-ibm-aix-xcoff -code-model=small \
; RUN: -stop-after=machine-cp -print-before=register-coalescer 2>&1 < \
; RUN: %s | FileCheck --check-prefix=SMALL %s

; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff -code-model=medium \
; RUN: -stop-after=machine-cp 2>&1 < %s | FileCheck --check-prefix=MEDIUM %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -code-model=large \
; RUN: -stop-after=machine-cp -print-before=register-coalescer 2>&1 < \
; RUN: %s | FileCheck --check-prefix=LARGE %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp \
; RUN: -print-before=register-coalescer 2>&1 < %s | FileCheck \
; RUN: --check-prefix=SMALL %s

@msg = common global ptr null, align 4
@ptr = common global ptr null, align 4

define void @foo() {
entry:
; SMALL: %0:gprc_and_gprc_nor0 = LWZtoc @msg, $r2 :: (load (s32) from got)
; SMALL: %1:gprc = LWZ 0, %0:gprc_and_gprc_nor0 :: (dereferenceable load (s32) from @msg)
; SMALL: %2:gprc_and_gprc_nor0 = LWZtoc @ptr, $r2 :: (load (s32) from got)
; SMALL: STW %1:gprc, 0, %2:gprc_and_gprc_nor0 :: (store (s32) into @ptr)

; MEDIUM: Medium code model is not supported on AIX.

; LARGE: %0:gprc_and_gprc_nor0 = ADDIStocHA $r2, @msg
; LARGE: %1:gprc_and_gprc_nor0 = LWZtocL @msg, %0:gprc_and_gprc_nor0, implicit $r2 :: (load (s32) from got)
; LARGE: %2:gprc = LWZ 0, %1:gprc_and_gprc_nor0 :: (dereferenceable load (s32) from @msg)
; LARGE: %3:gprc_and_gprc_nor0 = ADDIStocHA $r2, @ptr
; LARGE: %4:gprc_and_gprc_nor0 = LWZtocL @ptr, %3:gprc_and_gprc_nor0, implicit $r2 :: (load (s32) from got)
; LARGE: STW %2:gprc, 0, %4:gprc_and_gprc_nor0 :: (store (s32) into @ptr)

  %0 = load ptr, ptr @msg, align 4
  store ptr %0, ptr @ptr, align 4
  ret void
}
