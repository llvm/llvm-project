; RUN: llc < %s -mcpu=440 -mtriple=ppc32le-unknown-unknown | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc < %s -mcpu=440 -mtriple=ppc32-unknown-unknown   | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE

define i32 @foo(double %a) {
; CHECK-LABEL: foo:
; CHECK-DAG:            fctiwz [[FPR_1_i:[0-9]+]], {{[0-9]+}}
; CHECK-DAG:            stfd [[FPR_1_i]], [[#%u,VAL1_ADDR:]](1)
; CHECK-LE-DAG:         lwz {{[0-9]+}}, [[#%u,== VAL1_ADDR]](1)
; CHECK-BE-DAG:         lwz {{[0-9]+}}, [[#%u,== VAL1_ADDR + 4]](1)
; CHECK-DAG:            fctiwz [[FPR_2:[0-9]+]], {{[0-9]+}}
; CHECK-DAG:            stfd [[FPR_2]], [[#%u,VAL2_ADDR:]](1)
; CHECK-LE-DAG:         lwz {{[0-9]+}}, [[#%u,== VAL2_ADDR]](1)
; CHECK-BE-DAG:         lwz {{[0-9]+}}, [[#%u,== VAL2_ADDR + 4]](1)
entry:
        %tmp.1 = fptoui double %a to i32         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
