; RUN: llc < %s -mcpu=440 -mtriple=ppc32le-unknown-unknown | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc < %s -mcpu=440 -mtriple=ppc32-unknown-unknown   | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE

define i32 @foo(double %a) {
; CHECK-LABEL: foo:
; CHECK:                fctiwz [[FPR:[0-9]+]], {{[0-9]+}}
; CHECK:                stfd [[FPR]], [[#%u,VAL_ADDR:]](1)
; CHECK-LE:             lwz {{[0-9]+}}, [[#%u,== VAL_ADDR]](1)
; CHECK-BE:             lwz {{[0-9]+}}, [[#%u,== VAL_ADDR + 4]](1)
entry:
        %tmp.1 = fptoui double %a to i32         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
