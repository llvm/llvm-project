; RUN: opt < %s -passes=asan -S | FileCheck %s
target triple = "amdgcn-amd-amdhsa"

; Here we check that the global redzone sizes grow with the object size
; for objects in global address space.

@G10 = addrspace(1) global [10 x i8] zeroinitializer, align 1
; CHECK: @G10 = addrspace(1) global { [10 x i8], [22 x i8] }

@G31 = addrspace(1) global [31 x i8] zeroinitializer, align 1
@G32 = addrspace(1) global [32 x i8] zeroinitializer, align 1
@G33 = addrspace(1) global [33 x i8] zeroinitializer, align 1
; CHECK: @G31 = addrspace(1) global { [31 x i8], [33 x i8] }
; CHECK: @G32 = addrspace(1) global { [32 x i8], [32 x i8] }
; CHECK: @G33 = addrspace(1) global { [33 x i8], [63 x i8] }

@G63 = addrspace(1) global [63 x i8] zeroinitializer, align 1
@G64 = addrspace(1) global [64 x i8] zeroinitializer, align 1
@G65 = addrspace(1) global [65 x i8] zeroinitializer, align 1
; CHECK: @G63 = addrspace(1) global { [63 x i8], [33 x i8] }
; CHECK: @G64 = addrspace(1) global { [64 x i8], [32 x i8] }
; CHECK: @G65 = addrspace(1) global { [65 x i8], [63 x i8] }

@G127 = addrspace(1) global [127 x i8] zeroinitializer, align 1
@G128 = addrspace(1) global [128 x i8] zeroinitializer, align 1
@G129 = addrspace(1) global [129 x i8] zeroinitializer, align 1
; CHECK: @G127 = addrspace(1) global { [127 x i8], [33 x i8] }
; CHECK: @G128 = addrspace(1) global { [128 x i8], [32 x i8] }
; CHECK: @G129 = addrspace(1) global { [129 x i8], [63 x i8] }

@G255 = addrspace(1) global [255 x i8] zeroinitializer, align 1
@G256 = addrspace(1) global [256 x i8] zeroinitializer, align 1
@G257 = addrspace(1) global [257 x i8] zeroinitializer, align 1
; CHECK: @G255 = addrspace(1) global { [255 x i8], [33 x i8] }
; CHECK: @G256 = addrspace(1) global { [256 x i8], [64 x i8] }
; CHECK: @G257 = addrspace(1) global { [257 x i8], [95 x i8] }

@G511 = addrspace(1) global [511 x i8] zeroinitializer, align 1
@G512 = addrspace(1) global [512 x i8] zeroinitializer, align 1
@G513 = addrspace(1) global [513 x i8] zeroinitializer, align 1
; CHECK: @G511 = addrspace(1) global { [511 x i8], [97 x i8] }
; CHECK: @G512 = addrspace(1) global { [512 x i8], [128 x i8] }
; CHECK: @G513 = addrspace(1) global { [513 x i8], [159 x i8] }

@G1023 = addrspace(1) global [1023 x i8] zeroinitializer, align 1
@G1024 = addrspace(1) global [1024 x i8] zeroinitializer, align 1
@G1025 = addrspace(1) global [1025 x i8] zeroinitializer, align 1
; CHECK: @G1023 = addrspace(1) global { [1023 x i8], [225 x i8] }
; CHECK: @G1024 = addrspace(1) global { [1024 x i8], [256 x i8] }
; CHECK: @G1025 = addrspace(1) global { [1025 x i8], [287 x i8] }

@G1000000 = addrspace(1) global [1000000 x i8] zeroinitializer, align 1
@G10000000 = addrspace(1) global [10000000 x i8] zeroinitializer, align 1
@G100000000 = addrspace(1) global [100000000 x i8] zeroinitializer, align 1
; CHECK: @G1000000 = addrspace(1) global { [1000000 x i8], [249984 x i8] }
; CHECK: @G10000000 = addrspace(1) global { [10000000 x i8], [262144 x i8] }
; CHECK: @G100000000 = addrspace(1) global { [100000000 x i8], [262144 x i8] }
