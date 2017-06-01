; RUN: llvm-cat -o %t %s %S/Inputs/multi-module.ll
; RUN: not llvm-dis -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; ERROR: Expected a single module

; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; RUN: llvm-modextract -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=IR1 %s
; RUN: llvm-modextract -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=IR2 %s

; RUN: llvm-as -o %t1 %s
; RUN: llvm-as -o %t2 %S/Inputs/multi-module.ll
; RUN: llvm-cat -o %t %t1 %t2
; RUN: not llvm-dis -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; RUN: llvm-cat -b -o %t %t1 %t2
; RUN: not llvm-dis -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; RUN: llvm-modextract -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=IR1 %s
; RUN: llvm-modextract -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=IR2 %s

; RUN: llvm-cat -b -o %t3 %t %t
; RUN: not llvm-dis -o - %t3 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-bcanalyzer -dump %t3 | FileCheck --check-prefix=BCA4 %s

; RUN: llvm-modextract -n 0 -o - %t3 | llvm-dis | FileCheck --check-prefix=IR1 %s
; RUN: llvm-modextract -n 1 -o - %t3 | llvm-dis | FileCheck --check-prefix=IR2 %s
; RUN: llvm-modextract -n 2 -o - %t3 | llvm-dis | FileCheck --check-prefix=IR1 %s
; RUN: llvm-modextract -n 3 -o - %t3 | llvm-dis | FileCheck --check-prefix=IR2 %s

; BCA: <IDENTIFICATION_BLOCK
; BCA: <MODULE_BLOCK
; BCA: <IDENTIFICATION_BLOCK
; BCA: <MODULE_BLOCK

; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK
; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK
; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK
; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK

; IR1: define void @f1()
; IR2: define void @f2()

define void @f1() {
  ret void
}
