; RUN: llvm-offload-binary -o %t --image=file=%s,arch=abc,triple=x-y-z
; RUN: llvm-objdump --offloading %t | FileCheck %s
; RUN: llvm-offload-binary %t --image=file=%t2,arch=abc,triple=x-y-z
; RUN: echo "-o %t --image=file=%s,arch=abc,triple=x-y-z" > %t.rsp
; RUN: llvm-offload-binary @%t.rsp
; RUN: llvm-objdump --offloading %t | FileCheck %s
; RUN: diff %s %t2

;      CHECK: OFFLOADING IMAGE [0]:
; CHECK-NEXT: kind            <none>
; CHECK-NEXT: arch            abc
; CHECK-NEXT: triple          x-y-z
; CHECK-NEXT: producer        none

; RUN: llvm-offload-binary -o %t3 --image=file=%s
; RUN: llvm-offload-binary %t3 --image=file=%t4
; RUN: diff %s %t4

; Test nested OffloadBinary construction with multiple inner images.
; RUN: llvm-offload-binary -o %t5 --image=file=%s,arch=abc,triple=x-y-z --image=file=%s,arch=def,triple=x-y-z
; RUN: llvm-offload-binary -o %t6 --image=file=%t5,arch=nested,triple=x-y-z
; RUN: llvm-objdump --offloading %t6 | FileCheck %s --check-prefix=NESTED

; NESTED: OFFLOADING IMAGE [0]:
; NESTED: arch            nested
; NESTED: nested images   2
; NESTED:   OFFLOADING IMAGE [0.0]:
; NESTED:   OFFLOADING IMAGE [0.1]: