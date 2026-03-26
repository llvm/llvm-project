; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p1:16:16-p2:32:32-p3:64:64-i8:8-i32:32-i64:64"

; Too large for 16-bit address space.
@G1 = internal addrspace(1) global [65536 x i8] zeroinitializer, align 4

; Too large for 32-bit address space.
@G2 = internal addrspace(2) global [2147483648 x i16] zeroinitializer, align 4

; Fit within the address spaces
@G3 = internal addrspace(1) global [65535 x i8] zeroinitializer, align 4
@G4 = internal addrspace(2) global [2147483647 x i16] zeroinitializer, align 4

; CHECK: Global variable is too large to fit into the address space
; CHECK-NEXT: ptr addrspace(1) @G1
; CHECK-NEXT: [65536 x i8]
; CHECK: Global variable is too large to fit into the address space
; CHECK-NEXT: ptr addrspace(2) @G2
; CHECK-NEXT: [2147483648 x i16]
; CHECK-NOT: Global variable is too large to fit into the address space
