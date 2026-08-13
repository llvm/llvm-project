; UNSUPPORTED: system-zos
;
; Create a small shared object and ensure llvm-offload-binary does NOT extract
; offload images from a shared object (we expect packager to ignore .so files).
;
; RUN: echo 'int foo(){return 0;}' > %t.c
; RUN: %clang -shared -fPIC -o %t.so %t.c
; RUN: llvm-offload-binary -o %t.img --image=file=%t.so,arch=abc,triple=x-y-z\n+; RUN: llvm-objdump --offloading %t.img | FileCheck %s --check-prefix=NO-EXTRACT || true
;
; NO-EXTRACT-NOT: OFFLOADING IMAGE

