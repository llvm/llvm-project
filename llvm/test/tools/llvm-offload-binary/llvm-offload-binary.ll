; RUN: llvm-offload-binary -o %t --image=file=%s,arch=abc,triple=x-y-z
; RUN: llvm-objdump --offloading %t | FileCheck %s
; RUN: llvm-offload-binary %t --image=file=%t2,arch=abc,triple=x-y-z
; RUN: diff %s %t2

;      CHECK: OFFLOADING IMAGE [0]:
; CHECK-NEXT: kind            <none>
; CHECK-NEXT: arch            abc
; CHECK-NEXT: triple          x-y-z
; CHECK-NEXT: producer        none
