; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc-ibm-aix-xcoff \
; RUN:   -stop-after=finalize-isel -mattr=+use-ptrgl-helper < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc64-ibm-aix-xcoff \
; RUN:   -stop-after=finalize-isel -mattr=+use-ptrgl-helper < %s | \
; RUN:  FileCheck --check-prefix=CHECK64 %s

define i32 @has_strictfp(ptr noundef readonly captures(none) %fp) #0 {
entry:
  %call = tail call i32 %fp() strictfp
  ret i32 %call
}

attributes #0 = { strictfp }

; CHECK: BL_LWZinto_toc_RM &"._ptrgl[PR]", csr_aix32, implicit-def dead $lr, implicit-def dead $r2, implicit-def dead $rm, implicit $rm, implicit $r1, implicit $r11, implicit $r2, implicit-def $r1, implicit-def $r3

; CHECK64: BL8_LDinto_toc_RM &"._ptrgl[PR]", csr_ppc64, implicit-def dead $lr8, implicit-def dead $x2, implicit-def dead $rm, implicit $rm, implicit $x1, implicit $x11, implicit $x2, implicit-def $r1, implicit-def $x3
