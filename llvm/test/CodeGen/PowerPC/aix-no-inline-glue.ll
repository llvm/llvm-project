; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc-ibm-aix-xcoff \
; RUN:   -mattr=+use-ptrgl-helper < %s | FileCheck --check-prefixes=CHECK,CHECK32 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc64-ibm-aix-xcoff \
; RUN:   -mattr=+use-ptrgl-helper < %s | FileCheck --check-prefixes=CHECK,CHECK64 %s

; RUN: llc -stop-after=finalize-isel  -verify-machineinstrs -mcpu=pwr8 \
; RUN:   -mtriple powerpc-ibm-aix-xcoff -mattr=+use-ptrgl-helper < %s | \
; RUN:   FileCheck --check-prefix=MIR32 %s

; RUN: llc -stop-after=finalize-isel  -verify-machineinstrs -mcpu=pwr8 \
; RUN:   -mtriple powerpc64-ibm-aix-xcoff -mattr=+use-ptrgl-helper < %s | \
; RUN:   FileCheck --check-prefix=MIR64 %s

; RUN: not llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc-unknown-linux \
; RUN:   -mattr=+use-ptrgl-helper 2>&1 < %s | FileCheck --check-prefix=ERROR %s

; ERROR: use-ptrgl-helper feature is only supported on AIX

@a = dso_local global i32 55, align 4
@d = dso_local global double 3.141590e+00, align 8
@fp = dso_local global ptr null, align 8

define i32 @caller1(ptr noundef readonly captures(none) %fp) {
entry:
  %call = tail call i32 %fp(i32 signext 1, i32 signext 2, i32 signext 3)
  ret i32 %call
}

; CHECK-LABEL: .caller1
; CHECK-DAG:    mr 11, 3
; CHECK-DAG:    li 3, 1
; CHECK-DAG:    li 4, 2
; CHECK-DAG:    li 5, 3
; CHECK: bl ._ptrgl[PR]
; CHECK32-NEXT: lwz 2, 20(1)
; CHECK64-NEXT: ld 2, 40(1)

; MIR32: name:            caller1
; MIR32:   %0:gprc = COPY $r3
; MIR32:   ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; MIR32:   $r11 = COPY %0
; MIR32:   BL_LWZinto_toc &"._ptrgl[PR]", csr_aix32, implicit-def dead $lr, implicit-def dead $r2, implicit $rm, implicit $r1, implicit $r11, implicit $r3, implicit $r4, implicit $r5, implicit $r2, implicit-def $r1, implicit-def $r3
; MIR32:  ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; MIR64: name:            caller1
; MIR64:   %0:g8rc = COPY $x3
; MIR64:   ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; MIR64:   $x11 = COPY %0
; MIR64:   BL8_LDinto_toc &"._ptrgl[PR]", csr_ppc64, implicit-def dead $lr8, implicit-def dead $x2, implicit $rm, implicit $x1, implicit $x11, implicit $x3, implicit $x4, implicit $x5, implicit $x2, implicit-def $r1, implicit-def $x3
; MIR64:   ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define dso_local zeroext i1 @caller2() {
entry:
  %0 = load ptr, ptr @fp
  %1 = load i32, ptr @a
  %2 = load double, ptr @d
  %call = tail call zeroext i1 %0(i32 noundef signext %1, double noundef %2, ptr noundef nonnull @a)
  ret i1 %call
}

; CHECK-LABEL: .caller2
; CHECK64: ld [[REG:[0-9]+]], L..C{{[0-9]+}}(2)  # @fp
; CHECK64: ld 11, 0([[REG]])
; CHECK32: lwz [[REG:[0-9]+]], L..C{{[0-9]+}}(2) # @fp
; CHECK32: lwz 11, 0([[REG]])
; CHECK: bl ._ptrgl[PR]
; CHECK32-NEXT: lwz 2, 20(1)
; CHECK64-NEXT: ld 2, 40(1)

; MIR32: name:            caller2
; MIR32:   %0:gprc_and_gprc_nor0 = LWZtoc @fp, $r2 :: (load (s32) from got)
; MIR32:   %1:gprc = LWZ 0, killed %0 :: (dereferenceable load (s32) from @fp, align 8)
; MIR32:   ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; MIR32:   $r11 = COPY %1
; MIR32:   BL_LWZinto_toc &"._ptrgl[PR]", csr_aix32, implicit-def dead $lr, implicit-def dead $r2, implicit $rm, implicit $r1, implicit $r11, implicit $r3, implicit $f1, implicit $r6, implicit $r2, implicit-def $r1, implicit-def $r3
; MIR32:   ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; MIR64: name:            caller2
; MIR64:   %0:g8rc_and_g8rc_nox0 = LDtoc @fp, $x2 :: (load (s64) from got)
; MIR64:   %1:g8rc = LD 0, killed %0 :: (dereferenceable load (s64) from @fp)
; MIR64:   ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; MIR64:   $x11 = COPY %1
; MIR64:   BL8_LDinto_toc &"._ptrgl[PR]", csr_ppc64, implicit-def dead $lr8, implicit-def dead $x2, implicit $rm, implicit $x1, implicit $x11, implicit $x3, implicit $f1, implicit $x5, implicit $x2, implicit-def $r1, implicit-def $x3
; MIR64:   ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; CHECK: .extern ._ptrgl[PR]
