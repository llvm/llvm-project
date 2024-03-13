; RUN: llc -march=bpfel -mcpu=v4 < %s | FileCheck %s

; Generated from the following C code:
;
;   #define __as __attribute__((address_space(272)))
;   __as const char a[2] = {1,2};
;   __as char b[2] = {3,4};
;   __as char c[2];
;
; Using the following command:
;
;   clang --target=bpf -O2 -S -emit-llvm -o t.ll t.c


@a = dso_local local_unnamed_addr addrspace(272) constant [2 x i8] [i8 1, i8 2], align 1
@b = dso_local local_unnamed_addr addrspace(272) global [2 x i8] [i8 3, i8 4], align 1
@c = dso_local local_unnamed_addr addrspace(272) global [2 x i8] zeroinitializer, align 1

; Verify that a,b,c reside in the same section

; CHECK:     .section .address_space.272,"aw",@progbits
; CHECK-NOT: .section
; CHECK:     .globl  a
; CHECK:     .ascii  "\001\002"
; CHECK-NOT: .section
; CHECK:     .globl  b
; CHECK:     .ascii  "\003\004"
; CHECK-NOT: .section
; CHECK:     .globl  c
; CHECK:     .zero   2
