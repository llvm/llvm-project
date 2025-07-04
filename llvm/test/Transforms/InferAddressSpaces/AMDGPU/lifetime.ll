; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

define i32 @lifetime_flat_pointer() {
; CHECK-LABEL: define i32 @lifetime_flat_pointer() {
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK-NEXT:    call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) [[ALLOCA]])
; CHECK-NEXT:    store i32 1, ptr addrspace(5) [[ALLOCA]], align 4
; CHECK-NEXT:    %ret = load i32, ptr addrspace(5) [[ALLOCA]], align 4
; CHECK-NEXT:    call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) [[ALLOCA]])
; CHECK-NEXT:    ret i32 %ret
;
  %alloca = alloca i32, align 4, addrspace(5)
  %flat = addrspacecast ptr addrspace(5) %alloca to ptr
  call void @llvm.lifetime.start.p0(i64 4 , ptr %flat)
  store i32 1, ptr %flat, align 4
  %ret = load i32, ptr %flat, align 4
  call void @llvm.lifetime.end.p0(i64 4 , ptr %flat)
  ret i32 %ret
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
