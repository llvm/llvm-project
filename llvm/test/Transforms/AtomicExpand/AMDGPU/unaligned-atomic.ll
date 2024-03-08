; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=atomic-expand %s 2>&1 | FileCheck --check-prefix=GCN %s

define i32 @atomic_load_global_align1(ptr addrspace(1) %ptr) {
; GCN-LABEL: @atomic_load_global_align1(
; GCN-NEXT:    [[TMP2:%.*]] = addrspacecast ptr addrspace(1) [[PTR:%.*]] to ptr
; GCN-NEXT:    [[TMP3:%.*]] = alloca i32, align 4, addrspace(5)
; GCN-NEXT:    call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) [[TMP3]])
; GCN-NEXT:    call void @__atomic_load(i64 4, ptr [[TMP2]], ptr addrspace(5) [[TMP3]], i32 5)
; GCN-NEXT:    [[TMP5:%.*]] = load i32, ptr addrspace(5) [[TMP3]], align 4
; GCN-NEXT:    call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) [[TMP3]])
; GCN-NEXT:    ret i32 [[TMP5]]
;
  %val = load atomic i32, ptr addrspace(1) %ptr  seq_cst, align 1
  ret i32 %val
}

define void @atomic_store_global_align1(ptr addrspace(1) %ptr, i32 %val) {
; GCN-LABEL: @atomic_store_global_align1(
; GCN-NEXT:    [[TMP2:%.*]] = addrspacecast ptr addrspace(1) [[PTR:%.*]] to ptr
; GCN-NEXT:    [[TMP3:%.*]] = alloca i32, align 4, addrspace(5)
; GCN-NEXT:    call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) [[TMP3]])
; GCN-NEXT:    store i32 [[VAL:%.*]], ptr addrspace(5) [[TMP3]], align 4
; GCN-NEXT:    call void @__atomic_store(i64 4, ptr [[TMP2]], ptr addrspace(5) [[TMP3]], i32 0)
; GCN-NEXT:    call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) [[TMP3]])
; GCN-NEXT:    ret void
;
  store atomic i32 %val, ptr addrspace(1) %ptr monotonic, align 1
  ret void
}
