; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; CHECK-LABEL: @objectsize_group_to_flat_i32(
; CHECK: %val = call i32 @llvm.objectsize.i32.p3i8(i8 addrspace(3)* %group.ptr, i1 true, i1 false)
define i32 @objectsize_group_to_flat_i32(i8 addrspace(3)* %group.ptr) #0 {
  %cast = addrspacecast i8 addrspace(3)* %group.ptr to i8 addrspace(4)*
  %val = call i32 @llvm.objectsize.i32.p4i8(i8 addrspace(4)* %cast, i1 true, i1 false)
  ret i32 %val
}

; CHECK-LABEL: @objectsize_global_to_flat_i64(
; CHECK: %val = call i64 @llvm.objectsize.i64.p3i8(i8 addrspace(3)* %global.ptr, i1 true, i1 false)
define i64 @objectsize_global_to_flat_i64(i8 addrspace(3)* %global.ptr) #0 {
  %cast = addrspacecast i8 addrspace(3)* %global.ptr to i8 addrspace(4)*
  %val = call i64 @llvm.objectsize.i64.p4i8(i8 addrspace(4)* %cast, i1 true, i1 false)
  ret i64 %val
}

; CHECK-LABEL: @atomicinc_global_to_flat_i32(
; CHECK: call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %global.ptr, i32 %y, i32 0, i32 0, i1 false)
define i32 @atomicinc_global_to_flat_i32(i32 addrspace(1)* %global.ptr, i32 %y) #0 {
  %cast = addrspacecast i32 addrspace(1)* %global.ptr to i32 addrspace(4)*
  %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %cast, i32 %y, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicinc_group_to_flat_i32(
; CHECK: %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* %group.ptr, i32 %y, i32 0, i32 0, i1 false)
define i32 @atomicinc_group_to_flat_i32(i32 addrspace(3)* %group.ptr, i32 %y) #0 {
  %cast = addrspacecast i32 addrspace(3)* %group.ptr to i32 addrspace(4)*
  %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %cast, i32 %y, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicinc_global_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %global.ptr, i64 %y, i32 0, i32 0, i1 false)
define i64 @atomicinc_global_to_flat_i64(i64 addrspace(1)* %global.ptr, i64 %y) #0 {
  %cast = addrspacecast i64 addrspace(1)* %global.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @atomicinc_group_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.inc.i64.p3i64(i64 addrspace(3)* %group.ptr, i64 %y, i32 0, i32 0, i1 false)
define i64 @atomicinc_group_to_flat_i64(i64 addrspace(3)* %group.ptr, i64 %y) #0 {
  %cast = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @atomicdec_global_to_flat_i32(
; CHECK: call i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* %global.ptr, i32 %val, i32 0, i32 0, i1 false)
define i32 @atomicdec_global_to_flat_i32(i32 addrspace(1)* %global.ptr, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(1)* %global.ptr to i32 addrspace(4)*
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* %cast, i32 %val, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicdec_group_to_flat_i32(
; CHECK: %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p3i32(i32 addrspace(3)* %group.ptr, i32 %val, i32 0, i32 0, i1 false)
define i32 @atomicdec_group_to_flat_i32(i32 addrspace(3)* %group.ptr, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(3)* %group.ptr to i32 addrspace(4)*
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* %cast, i32 %val, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK-LABEL: @atomicdec_global_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.dec.i64.p1i64(i64 addrspace(1)* %global.ptr, i64 %y, i32 0, i32 0, i1 false)
define i64 @atomicdec_global_to_flat_i64(i64 addrspace(1)* %global.ptr, i64 %y) #0 {
  %cast = addrspacecast i64 addrspace(1)* %global.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @atomicdec_group_to_flat_i64(
; CHECK: call i64 @llvm.amdgcn.atomic.dec.i64.p3i64(i64 addrspace(3)* %group.ptr, i64 %y, i32 0, i32 0, i1 false
define i64 @atomicdec_group_to_flat_i64(i64 addrspace(3)* %group.ptr, i64 %y) #0 {
  %cast = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK-LABEL: @volatile_atomicinc_group_to_flat_i64(
; CHECK-NEXT: %1 = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
; CHECK-NEXT: %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %1, i64 %y, i32 0, i32 0, i1 true)
define i64 @volatile_atomicinc_group_to_flat_i64(i64 addrspace(3)* %group.ptr, i64 %y) #0 {
  %cast = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 true)
  ret i64 %ret
}

; CHECK-LABEL: @volatile_atomicdec_global_to_flat_i32(
; CHECK-NEXT: %1 = addrspacecast i32 addrspace(1)* %global.ptr to i32 addrspace(4)*
; CHECK-NEXT: %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* %1, i32 %val, i32 0, i32 0, i1 true)
define i32 @volatile_atomicdec_global_to_flat_i32(i32 addrspace(1)* %global.ptr, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(1)* %global.ptr to i32 addrspace(4)*
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* %cast, i32 %val, i32 0, i32 0, i1 true)
  ret i32 %ret
}

; CHECK-LABEL: @volatile_atomicdec_group_to_flat_i32(
; CHECK-NEXT: %1 = addrspacecast i32 addrspace(3)* %group.ptr to i32 addrspace(4)*
; CHECK-NEXT: %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* %1, i32 %val, i32 0, i32 0, i1 true)
define i32 @volatile_atomicdec_group_to_flat_i32(i32 addrspace(3)* %group.ptr, i32 %val) #0 {
  %cast = addrspacecast i32 addrspace(3)* %group.ptr to i32 addrspace(4)*
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* %cast, i32 %val, i32 0, i32 0, i1 true)
  ret i32 %ret
}

; CHECK-LABEL: @volatile_atomicdec_global_to_flat_i64(
; CHECK-NEXT: %1 = addrspacecast i64 addrspace(1)* %global.ptr to i64 addrspace(4)*
; CHECK-NEXT: %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p4i64(i64 addrspace(4)* %1, i64 %y, i32 0, i32 0, i1 true)
define i64 @volatile_atomicdec_global_to_flat_i64(i64 addrspace(1)* %global.ptr, i64 %y) #0 {
  %cast = addrspacecast i64 addrspace(1)* %global.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 true)
  ret i64 %ret
}

; CHECK-LABEL: @volatile_atomicdec_group_to_flat_i64(
; CHECK-NEXT: %1 = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
; CHECK-NEXT: %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p4i64(i64 addrspace(4)* %1, i64 %y, i32 0, i32 0, i1 true)
define i64 @volatile_atomicdec_group_to_flat_i64(i64 addrspace(3)* %group.ptr, i64 %y) #0 {
  %cast = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 true)
  ret i64 %ret
}

; CHECK-LABEL: @invalid_variable_volatile_atomicinc_group_to_flat_i64(
; CHECK-NEXT: %1 = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
; CHECK-NEXT: %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %1, i64 %y, i32 0, i32 0, i1 %volatile.var)
define i64 @invalid_variable_volatile_atomicinc_group_to_flat_i64(i64 addrspace(3)* %group.ptr, i64 %y, i1 %volatile.var) #0 {
  %cast = addrspacecast i64 addrspace(3)* %group.ptr to i64 addrspace(4)*
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %cast, i64 %y, i32 0, i32 0, i1 %volatile.var)
  ret i64 %ret
}

declare i32 @llvm.objectsize.i32.p4i8(i8 addrspace(4)*, i1, i1) #1
declare i64 @llvm.objectsize.i64.p4i8(i8 addrspace(4)*, i1, i1) #1
declare i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* nocapture, i32, i32, i32, i1) #2
declare i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* nocapture, i64, i32, i32, i1) #2
declare i32 @llvm.amdgcn.atomic.dec.i32.p4i32(i32 addrspace(4)* nocapture, i32, i32, i32, i1) #2
declare i64 @llvm.amdgcn.atomic.dec.i64.p4i64(i64 addrspace(4)* nocapture, i64, i32, i32, i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind argmemonly }
