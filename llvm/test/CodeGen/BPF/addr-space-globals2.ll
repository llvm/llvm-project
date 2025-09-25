; RUN: llc -mtriple=bpfel -mcpu=v4 < %s | FileCheck %s

; Generated from the following C code:
;
;   __attribute__((address_space(1))) char a[2] = {1,2};
;   __attribute__((address_space(2))) char b[2] = {3,4};
;
; Using the following command:
;
;   clang --target=bpf -O2 -S -emit-llvm -o t.ll t.c

@a = dso_local local_unnamed_addr addrspace(1) global [2 x i8] [i8 1, i8 2], align 1
@b = dso_local local_unnamed_addr addrspace(2) global [2 x i8] [i8 3, i8 4], align 1

; Verify that a,b reside in separate sections

; CHECK:     .section .addr_space.1,"aw",@progbits
; CHECK-NOT: .section
; CHECK:     .globl  a
; CHECK:     .ascii  "\001\002"

; CHECK:     .section .addr_space.2,"aw",@progbits
; CHECK-NOT: .section
; CHECK:     .globl  b
; CHECK:     .ascii  "\003\004"
