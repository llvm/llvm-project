; RUN: opt -S -passes=infer-address-spaces %s | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

define i32 @lifetime_flat_pointer() {
; CHECK-LABEL: define i32 @lifetime_flat_pointer() {
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr [[ALLOCA]] to ptr addrspace(5)
; CHECK-NEXT:    call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) [[TMP1]])
; CHECK-NEXT:    store i32 1, ptr addrspace(5) [[TMP1]], align 4
; CHECK-NEXT:    %ret = load i32, ptr addrspace(5) [[TMP1]], align 4
; CHECK-NEXT:    call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) [[TMP1]])
; CHECK-NEXT:    ret i32 %ret
;
  %alloca = alloca i32, align 4
  %1 = addrspacecast ptr %alloca to ptr addrspace(5)
  %2 = addrspacecast ptr addrspace(5) %1 to ptr
  %3 = addrspacecast ptr addrspace(5) %1 to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %2)
  store i32 1, ptr addrspace(5) %1, align 4
  %ret = load i32, ptr addrspace(5) %1, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr %3)
  ret i32 %ret
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
