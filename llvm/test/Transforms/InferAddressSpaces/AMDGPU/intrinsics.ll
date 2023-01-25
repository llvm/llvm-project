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

; CHECK-LABEL: @atomicinc_global_to_flat_i32(
; CHECK: call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %global.ptr, i32 %y, i32 0, i32 0, i1 false)
define i32 @atomicinc_global_to_flat_i32(ptr addrspace(1) %global.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %cast, i32 %y, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicinc_group_to_flat_i32(
; CHECK: %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %group.ptr, i32 %y, i32 0, i32 0, i1 false)
define i32 @atomicinc_group_to_flat_i32(ptr addrspace(3) %group.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %cast, i32 %y, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicinc_global_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %global.ptr, i64 %y, i32 0, i32 0, i1 false)
define i64 @atomicinc_global_to_flat_i64(ptr addrspace(1) %global.ptr, i64 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @atomicinc_group_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %group.ptr, i64 %y, i32 0, i32 0, i1 false)
define i64 @atomicinc_group_to_flat_i64(ptr addrspace(3) %group.ptr, i64 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @atomicdec_global_to_flat_i32(
; CHECK: call i32 @llvm.amdgcn.atomic.dec.i32.p1(ptr addrspace(1) %global.ptr, i32 %val, i32 0, i32 0, i1 false)
define i32 @atomicdec_global_to_flat_i32(ptr addrspace(1) %global.ptr, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %cast, i32 %val, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicdec_group_to_flat_i32(
; CHECK: %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p3(ptr addrspace(3) %group.ptr, i32 %val, i32 0, i32 0, i1 false)
define i32 @atomicdec_group_to_flat_i32(ptr addrspace(3) %group.ptr, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %cast, i32 %val, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicdec_global_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.dec.i64.p1(ptr addrspace(1) %global.ptr, i64 %y, i32 0, i32 0, i1 false)
define i64 @atomicdec_global_to_flat_i64(ptr addrspace(1) %global.ptr, i64 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @atomicdec_group_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.dec.i64.p3(ptr addrspace(3) %group.ptr, i64 %y, i32 0, i32 0, i1 false
define i64 @atomicdec_group_to_flat_i64(ptr addrspace(3) %group.ptr, i64 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @volatile_atomicinc_group_to_flat_i64(
; CHECK-NEXT: %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
; CHECK-NEXT: %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 true)
define i64 @volatile_atomicinc_group_to_flat_i64(ptr addrspace(3) %group.ptr, i64 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 true)
  ret i64 %ret
}

; CHECK-LABEL: @volatile_atomicdec_global_to_flat_i32(
; CHECK-NEXT: %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
; CHECK-NEXT: %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %cast, i32 %val, i32 0, i32 0, i1 true)
define i32 @volatile_atomicdec_global_to_flat_i32(ptr addrspace(1) %global.ptr, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %cast, i32 %val, i32 0, i32 0, i1 true)
  ret i32 %ret
}

; CHECK-LABEL: @volatile_atomicdec_group_to_flat_i32(
; CHECK-NEXT: %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
; CHECK-NEXT: %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %cast, i32 %val, i32 0, i32 0, i1 true)
define i32 @volatile_atomicdec_group_to_flat_i32(ptr addrspace(3) %group.ptr, i32 %val) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %cast, i32 %val, i32 0, i32 0, i1 true)
  ret i32 %ret
}

; CHECK-LABEL: @volatile_atomicdec_global_to_flat_i64(
; CHECK-NEXT: %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
; CHECK-NEXT: %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 true)
define i64 @volatile_atomicdec_global_to_flat_i64(ptr addrspace(1) %global.ptr, i64 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 true)
  ret i64 %ret
}

; CHECK-LABEL: @volatile_atomicdec_group_to_flat_i64(
; CHECK-NEXT: %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
; CHECK-NEXT: %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 true)
define i64 @volatile_atomicdec_group_to_flat_i64(ptr addrspace(3) %group.ptr, i64 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %cast, i64 %y, i32 0, i32 0, i1 true)
  ret i64 %ret
}

declare i32 @llvm.objectsize.i32.p0(ptr, i1, i1, i1) #1
declare i64 @llvm.objectsize.i64.p0(ptr, i1, i1, i1) #1
declare i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr nocapture, i32, i32, i32, i1) #2
declare i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr nocapture, i64, i32, i32, i1) #2
declare i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr nocapture, i32, i32, i32, i1) #2
declare i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr nocapture, i64, i32, i32, i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind argmemonly }
