; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; CHECK-LABEL: @objectsize_group_to_flat_i32(
; CHECK: %val = call i32 @llvm.objectsize.i32.p3(ptr addrspace(3) %group.ptr, i1 true, i1 false, i1 false)
define i32 @objectsize_group_to_flat_i32(ptr addrspace(3) %group.ptr) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %val = call i32 @llvm.objectsize.i32.p0(ptr %cast, i1 true, i1 false, i1 false)
  ret i32 %val
}

; CHECK-LABEL: @objectsize_global_to_flat_i64(
; CHECK: %val = call i64 @llvm.objectsize.i64.p3(ptr addrspace(3) %global.ptr, i1 true, i1 false, i1 false)
define i64 @objectsize_global_to_flat_i64(ptr addrspace(3) %global.ptr) #0 {
  %cast = addrspacecast ptr addrspace(3) %global.ptr to ptr
  %val = call i64 @llvm.objectsize.i64.p0(ptr %cast, i1 true, i1 false, i1 false)
  ret i64 %val
}

declare i32 @llvm.objectsize.i32.p0(ptr, i1, i1, i1) #1
declare i64 @llvm.objectsize.i64.p0(ptr, i1, i1, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind argmemonly }
