; On z/OS, ebcdic output from printf is converted into utf-8 as
; the output being redirected, causing the magic bytes to be incorrect.
; TODO: use a builtin version of printf
; UNSUPPORTED: system-zos

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

; Test extracting all images without specifying --image filters.
; RUN: rm -rf %t.dir && mkdir %t.dir
; RUN: cd %t.dir
; RUN: llvm-offload-binary %t | FileCheck --check-prefix=EXTRACT %s

; EXTRACT: Extracted: llvm-offload-binary.{{.*}}-x-y-z-abc.0.

; Test nested OffloadBinary construction with multiple inner images.
; RUN: llvm-offload-binary -o %t5 --image=file=%s,arch=abc,triple=x-y-z --image=file=%s,arch=def,triple=x-y-z
; RUN: llvm-offload-binary -o %t6 --image=file=%t5,arch=nested,triple=x-y-z
; RUN: llvm-objdump --offloading %t6 | FileCheck %s --check-prefix=NESTED

; NESTED:        OFFLOADING IMAGE [0]:
; NESTED-NEXT:   kind            <none>
; NESTED-NEXT:   arch            nested
; NESTED-NEXT:   triple          x-y-z
; NESTED-NEXT:   producer        none
; NESTED-NEXT:   image size      {{.*}} bytes
; NESTED-NEXT:   nested images   2
; NESTED-EMPTY:
; NESTED-NEXT:     OFFLOADING IMAGE [0.0]:
; NESTED-NEXT:     kind            <none>
; NESTED-NEXT:     arch            abc
; NESTED-NEXT:     triple          x-y-z
; NESTED-NEXT:     producer        none
; NESTED-NEXT:     image size      {{.*}} bytes
; NESTED-EMPTY:
; NESTED-NEXT:     OFFLOADING IMAGE [0.1]:
; NESTED-NEXT:     kind            <none>
; NESTED-NEXT:     arch            def
; NESTED-NEXT:     triple          x-y-z
; NESTED-NEXT:     producer        none
; NESTED-NEXT:     image size      {{.*}} bytes

; Test complex nested OffloadBinary construction with multiple levels.
; RUN: llvm-offload-binary -o %t7 --image=file=%s,arch=abc,triple=x-y-z --image=file=%t5,arch=nested,triple=x-y-z
; RUN: llvm-offload-binary -o %t8 --image=file=%t7,arch=nested,triple=x-y-z --image=file=%t5,arch=nested2,triple=x-y-z
; RUN: llvm-objdump --offloading %t8 | FileCheck %s --check-prefix=NESTED2

; NESTED2:        OFFLOADING IMAGE [0]:
; NESTED2-NEXT:   kind            <none>
; NESTED2-NEXT:   arch            nested
; NESTED2-NEXT:   triple          x-y-z
; NESTED2-NEXT:   producer        none
; NESTED2-NEXT:   image size      {{.*}} bytes
; NESTED2-NEXT:   nested images   2
; NESTED2-EMPTY:
; NESTED2-NEXT:   OFFLOADING IMAGE [0.0]:
; NESTED2-NEXT:     kind            <none>
; NESTED2-NEXT:     arch            abc
; NESTED2-NEXT:     triple          x-y-z
; NESTED2-NEXT:     producer        none
; NESTED2-NEXT:     image size      {{.*}} bytes
; NESTED2-EMPTY:
; NESTED2-NEXT:   OFFLOADING IMAGE [0.1]:
; NESTED2-NEXT:     kind            <none>
; NESTED2-NEXT:     arch            nested
; NESTED2-NEXT:     triple          x-y-z
; NESTED2-NEXT:     producer        none
; NESTED2-NEXT:     image size      {{.*}} bytes
; NESTED2-NEXT:     nested images   2
; NESTED2-EMPTY:
; NESTED2-NEXT:     OFFLOADING IMAGE [0.1.0]:
; NESTED2-NEXT:     kind            <none>
; NESTED2-NEXT:     arch            abc
; NESTED2-NEXT:     triple          x-y-z
; NESTED2-NEXT:     producer        none
; NESTED2-NEXT:     image size      {{.*}} bytes
; NESTED2-EMPTY:
; NESTED2-NEXT:     OFFLOADING IMAGE [0.1.1]:
; NESTED2-NEXT:     kind            <none>
; NESTED2-NEXT:     arch            def
; NESTED2-NEXT:     triple          x-y-z
; NESTED2-NEXT:     producer        none
; NESTED2-NEXT:     image size      {{.*}} bytes
; NESTED2-EMPTY:
; NESTED2-NEXT:   OFFLOADING IMAGE [1]:
; NESTED2-NEXT:     kind            <none>
; NESTED2-NEXT:     arch            nested2
; NESTED2-NEXT:     triple          x-y-z
; NESTED2-NEXT:     producer        none
; NESTED2-NEXT:     image size      {{.*}} bytes
; NESTED2-NEXT:     nested images   2
; NESTED2-EMPTY:
; NESTED2-NEXT:     OFFLOADING IMAGE [1.0]:
; NESTED2-NEXT:     kind            <none>
; NESTED2-NEXT:     arch            abc
; NESTED2-NEXT:     triple          x-y-z
; NESTED2-NEXT:     producer        none
; NESTED2-NEXT:     image size      {{.*}} bytes
; NESTED2-EMPTY:
; NESTED2-NEXT:     OFFLOADING IMAGE [1.1]:
; NESTED2-NEXT:     kind            <none>
; NESTED2-NEXT:     arch            def
; NESTED2-NEXT:     triple          x-y-z
; NESTED2-NEXT:     producer        none
; NESTED2-NEXT:     image size      {{.*}} bytes

; Test extracting nested images.
; RUN: llvm-offload-binary %t6 | FileCheck --check-prefix=EXTRACT-NESTED %s

; EXTRACT-NESTED:      Extracted: llvm-offload-binary.{{.*}}-x-y-z-abc.0.
; EXTRACT-NESTED-NEXT: Extracted: llvm-offload-binary.{{.*}}-x-y-z-def.1.

; Test mixed nested and non-nested images.
; RUN: llvm-offload-binary -o %t7 --image=file=%t5,arch=nested,triple=x-y-z --image=file=%s,arch=ghi,triple=x-y-z
; RUN: llvm-offload-binary %t7 | FileCheck --check-prefix=EXTRACT-MIXED %s

; EXTRACT-MIXED:      Extracted: llvm-offload-binary.{{.*}}-x-y-z-abc.0.
; EXTRACT-MIXED-NEXT: Extracted: llvm-offload-binary.{{.*}}-x-y-z-def.1.
; EXTRACT-MIXED-NEXT: Extracted: llvm-offload-binary.{{.*}}-x-y-z-ghi.2.

; Test extracting inner OffloadBinary with --image filter.
; RUN: llvm-offload-binary %t7 --image=file=%t8,arch=nested,triple=x-y-z
; RUN: diff %t5 %t8
