; RUN: llc -stop-after=finalize-isel  -verify-machineinstrs -mcpu=pwr8 \
; RUN:   -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -stop-after=finalize-isel  -verify-machineinstrs -mcpu=pwr8 \
; RUN:   -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK64

define i32 @OutOfLine(ptr noundef readonly captures(none) %fp) #0 {
entry:
  %call = tail call i32 %fp()
  ret i32 %call
}

define i32 @InLine(ptr noundef readonly captures(none) %fp) #1 {
entry:
  %call = tail call i32 %fp()
  ret i32 %call
}

attributes #0 = {"target-features"="+use-ptrgl-helper"}
attributes #1 = {"target-features"="-use-ptrgl-helper"}

; CHECK: name:            OutOfLine
; CHECK:  BL_LWZinto_toc &"._ptrgl[PR]", csr_aix32, implicit-def dead $lr, implicit-def dead $r2, implicit $rm, implicit $r1, implicit $r11, implicit $r2, implicit-def $r1, implicit-def $r3
; CHECK: name:            InLine
; CHECK: BCTRL_LWZinto_toc 20, $r1, csr_aix32, implicit-def dead $lr, implicit-def dead $r2, implicit $ctr, implicit $rm, implicit $r11, implicit $r2, implicit-def $r1, implicit-def $r3

; CHECK64: name:            OutOfLine
; CHECK64:  BL8_LDinto_toc &"._ptrgl[PR]", csr_ppc64, implicit-def dead $lr8, implicit-def dead $x2, implicit $rm, implicit $x1, implicit $x11, implicit $x2, implicit-def $r1, implicit-def $x3
; CHECK64: name:            InLine
; CHECK64:   BCTRL8_LDinto_toc 40, $x1, csr_ppc64, implicit-def dead $lr8, implicit-def dead $x2, implicit $ctr8, implicit $rm, implicit $x11, implicit $x2, implicit-def $r1, implicit-def $x3
