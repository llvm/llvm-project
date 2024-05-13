; RUN: opt -passes=amdgpu-clone-module-lds %s -S -o - | FileCheck %s

; RUN: opt -passes=amdgpu-clone-module-lds %s -S -o %t
; RUN: llvm-split -o %t %t -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=MOD0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=MOD1 %s

target triple = "amdgcn-amd-amdhsa"

; Before transformation,                    After transformation,
;  K1  K2    K3                              K1  K2    K3
;  |  /      |                               |  /      |
;  | /       |                               | /       |
;  A --------+               ==>             A --------+
;  |                                         |
;  |                                         |
;  B                                         B
;  |                                       / | \
;  X                                      X1 X2 X3
;  |                                      \  |  /
;  D                                       \ | /
;                                            D
; where X contains an LDS reference

; CHECK: [[GV_CLONE_0:@.*]] = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 16
; CHECK: [[GV_CLONE_1:@.*]] = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 16
; CHECK: [[GV:@.*]] = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 16
@lds_gv = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 16

define protected amdgpu_kernel void @kernel1(i32 %n) {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel1(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @A(i32 [[N]])
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @A(i32 %n)
  ret void
}

define protected amdgpu_kernel void @kernel2(i32 %n) {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel2(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @A(i32 [[N]])
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @A(i32 %n)
  ret void
}

define protected amdgpu_kernel void @kernel3(i32 %n) {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel3(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @A(i32 [[N]])
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @A(i32 %n)
  ret void
}

define void @A() {
; CHECK-LABEL: define void @A() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @B()
; CHECK-NEXT:    ret void
;
entry:
  call void @B()
  ret void
}

define i32 @B() {
; CHECK-LABEL: define i32 @B() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P:%.*]] = alloca i32, align 4
; CHECK-NEXT:    store i32 5, ptr [[P]], align 4
; CHECK-NEXT:    [[RET:%.*]] = call i32 @X(ptr [[P]])
; CHECK-NEXT:    [[RET_CLONE_1:%.*]] = call i32 @X.clone.1(ptr [[P]])
; CHECK-NEXT:    [[RET_CLONE_0:%.*]] = call i32 @X.clone.0(ptr [[P]])
; CHECK-NEXT:    ret i32 [[RET]]
;
entry:
  %p = alloca i32
  store i32 5, ptr %p
  %ret = call i32 @X(ptr %p)
  ret i32 %ret
}

define i32 @X(ptr %x) {
; CHECK-LABEL: define i32 @X(
; CHECK-SAME: ptr [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P:%.*]] = getelementptr inbounds [64 x i32], ptr addrspacecast (ptr addrspace(3) [[GV]] to ptr), i64 0, i64 0
; CHECK-NEXT:    [[V:%.*]] = load i32, ptr [[X]], align 4
; CHECK-NEXT:    call void @D(ptr [[P]])
; CHECK-NEXT:    store i32 [[V]], ptr [[P]], align 4
; CHECK-NEXT:    ret i32 [[V]]
;
entry:
  %p = getelementptr inbounds [64 x i32], ptr addrspacecast (ptr addrspace(3) @lds_gv to ptr), i64 0, i64 0
  %v = load i32, ptr %x
  call void @D(ptr %p)
  store i32 %v, ptr %p
  ret i32 %v
}

define void @D(ptr %x) {
; CHECK-LABEL: define void @D(ptr %x) {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     store i32 8, ptr %x, align 4
; CHECK-NEXT:     ret void
entry:
  store i32 8, ptr %x
  ret void
}

; CHECK-LABEL: define i32 @X.clone.0(ptr %x) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %p = getelementptr inbounds [64 x i32], ptr addrspacecast (ptr addrspace(3) [[GV_CLONE_0]] to ptr), i64 0, i64 0
; CHECK-NEXT:   %v = load i32, ptr %x, align 4
; CHECK-NEXT:   call void @D(ptr [[P]])
; CHECK-NEXT:   store i32 %v, ptr %p, align 4
; CHECK-NEXT:   ret i32 %v

; CHECK-LABEL: define i32 @X.clone.1(ptr %x) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %p = getelementptr inbounds [64 x i32], ptr addrspacecast (ptr addrspace(3) [[GV_CLONE_1]] to ptr), i64 0, i64 0
; CHECK-NEXT:   %v = load i32, ptr %x, align 4
; CHECK-NEXT:   call void @D(ptr [[P]])
; CHECK-NEXT:   store i32 %v, ptr %p, align 4
; CHECK-NEXT:   ret i32 %v

; MOD0: {{.*}} addrspace(3) global [64 x i32] undef, align 16
; MOD0: define i32 @X(ptr %x)

; MOD1: {{.*}} addrspace(3) global [64 x i32] poison, align 16
; MOD1: {{.*}} addrspace(3) global [64 x i32] poison, align 16
; MOD1: define protected amdgpu_kernel void @kernel1(i32 %n)
; MOD1: define protected amdgpu_kernel void @kernel2(i32 %n)
; MOD1: define protected amdgpu_kernel void @kernel3(i32 %n)
; MOD1: define void @A()
; MOD1: define i32 @B()
; MOD1: define i32 @X.clone.0(ptr %x)
