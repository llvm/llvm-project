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
; NESTED:   arch            abc
; NESTED:   OFFLOADING IMAGE [0.1]:
; NESTED:   arch            def

; Test complex nested OffloadBinary construction with multiple levels.
; RUN: llvm-offload-binary -o %t7 --image=file=%s,arch=abc,triple=x-y-z --image=file=%t5,arch=nested,triple=x-y-z
; RUN: llvm-offload-binary -o %t8 --image=file=%t7,arch=nested,triple=x-y-z --image=file=%t5,arch=nested2,triple=x-y-z
; RUN: llvm-objdump --offloading %t8 | FileCheck %s --check-prefix=NESTED2

; NESTED2: OFFLOADING IMAGE [0]:
; NESTED2: arch            nested
; NESTED2: nested images   2
; NESTED2:   OFFLOADING IMAGE [0.0]:
; NESTED2:   arch            abc
; NESTED2:   OFFLOADING IMAGE [0.1]:
; NESTED2:   arch            nested
; NESTED2:   nested images   2
; NESTED2:     OFFLOADING IMAGE [0.1.0]:
; NESTED2:     arch            abc
; NESTED2:     OFFLOADING IMAGE [0.1.1]:
; NESTED2:     arch            def
; NESTED2: OFFLOADING IMAGE [1]:
; NESTED2: arch            nested2
; NESTED2: nested images   2
; NESTED2:   OFFLOADING IMAGE [1.0]:
; NESTED2:   arch            abc
; NESTED2:   OFFLOADING IMAGE [1.1]:
; NESTED2:   arch            def
