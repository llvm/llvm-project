; Test the IR memory-effects contract of the fabric put intrinsics.
;
; RUN: opt -passes=aa-eval -aa-pipeline=nvptx-aa,basic-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AA
; RUN: opt -passes=aa-eval -aa-pipeline=basic-aa,nvptx-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AA
; RUN: opt -passes=gvn -aa-pipeline=nvptx-aa,basic-aa -S < %s | FileCheck %s --check-prefix=GVN
; RUN: opt -passes=gvn -aa-pipeline=basic-aa,nvptx-aa -S < %s | FileCheck %s --check-prefix=GVN
; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=ATTR

target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p6:32:32-p8:128:128-ni:8-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

declare ptr addrspace(8) @llvm.nvvm.fabric.handle_pair(i32, i64)
declare void @llvm.nvvm.fabric.try_put(ptr addrspace(8), ptr addrspace(3), ptr addrspace(3), i32, i16, i64, i1, i1, i32)
declare void @llvm.nvvm.fabric.try_put.counted_writes(ptr addrspace(8), ptr addrspace(8), ptr addrspace(3), ptr addrspace(3), i32, i64, i1, i32)

define i32 @put_effects(ptr addrspace(1) %global, i32 %id, i64 %offset,
                        ptr addrspace(3) %src, ptr addrspace(3) %bar) {
; AA-LABEL: Function: put_effects:
; The constructor stays memory-free.
; AA-DAG: NoModRef:  Ptr: {{.*}}%global{{[[:space:]]*}}<->{{[[:space:]]*}}{{.*}}@llvm.nvvm.fabric.handle_pair
; The put may access global memory through the handle.
; AA-DAG: Both ModRef:  Ptr: {{.*}}%global{{[[:space:]]*}}<->{{[[:space:]]*}}call void @llvm.nvvm.fabric.try_put(
; GVN-LABEL: define i32 @put_effects(
; GVN: %before = load i32, ptr addrspace(1) %global
; GVN: %after = load i32, ptr addrspace(1) %global
; GVN: %difference = sub i32 %after, %before
  %handle = call ptr addrspace(8) @llvm.nvvm.fabric.handle_pair(i32 %id, i64 %offset)
  %before = load i32, ptr addrspace(1) %global
  call void @llvm.nvvm.fabric.try_put(ptr addrspace(8) %handle, ptr addrspace(3) %src, ptr addrspace(3) %bar,
                                      i32 16, i16 0, i64 0, i1 false, i1 false, i32 0)
  %after = load i32, ptr addrspace(1) %global
  %difference = sub i32 %after, %before
  ret i32 %difference
}

; Both counted handles share one endpoint ID and use distinct offsets.
define i32 @counted_put_effects(ptr addrspace(1) %global, i32 %id,
                                i64 %data_offset, i64 %counter_offset,
                                ptr addrspace(3) %src, ptr addrspace(3) %bar) {
; AA-LABEL: Function: counted_put_effects:
; AA-DAG: NoModRef:  Ptr: {{.*}}%global{{[[:space:]]*}}<->{{[[:space:]]*}}{{.*}}@llvm.nvvm.fabric.handle_pair
; AA-DAG: Both ModRef:  Ptr: {{.*}}%global{{[[:space:]]*}}<->{{[[:space:]]*}}call void @llvm.nvvm.fabric.try_put.counted_writes(
; GVN-LABEL: define i32 @counted_put_effects(
; GVN: %before = load i32, ptr addrspace(1) %global
; GVN: %after = load i32, ptr addrspace(1) %global
; GVN: %difference = sub i32 %after, %before
  %data_handle = call ptr addrspace(8) @llvm.nvvm.fabric.handle_pair(i32 %id, i64 %data_offset)
  %counter_handle = call ptr addrspace(8) @llvm.nvvm.fabric.handle_pair(i32 %id, i64 %counter_offset)
  %before = load i32, ptr addrspace(1) %global
  call void @llvm.nvvm.fabric.try_put.counted_writes(ptr addrspace(8) %data_handle, ptr addrspace(8) %counter_handle,
                                                     ptr addrspace(3) %src, ptr addrspace(3) %bar, i32 16, i64 0, i1 false, i32 0)
  %after = load i32, ptr addrspace(1) %global
  %difference = sub i32 %after, %before
  ret i32 %difference
}

; The handle arrives as an argument; the resource access is based on it.
define i32 @put_handle_argument_effects(ptr addrspace(8) %handle,
                                        ptr addrspace(1) %global,
                                        ptr addrspace(3) %src,
                                        ptr addrspace(3) %bar) {
; AA-LABEL: Function: put_handle_argument_effects:
; AA: Both ModRef:  Ptr: {{.*}}%global{{[[:space:]]*}}<->{{[[:space:]]*}}call void @llvm.nvvm.fabric.try_put(
; GVN-LABEL: define i32 @put_handle_argument_effects(
; GVN: %before = load i32, ptr addrspace(1) %global
; GVN: %after = load i32, ptr addrspace(1) %global
; GVN: %difference = sub i32 %after, %before
  %before = load i32, ptr addrspace(1) %global
  call void @llvm.nvvm.fabric.try_put(ptr addrspace(8) %handle, ptr addrspace(3) %src, ptr addrspace(3) %bar, i32 16,
                                      i16 0, i64 0, i1 false, i1 false, i32 0)
  %after = load i32, ptr addrspace(1) %global
  %difference = sub i32 %after, %before
  ret i32 %difference
}

; A locally constructed handle versus an ordinary generic (AS0) pointer.
define i32 @put_generic_effects(ptr %generic, i32 %id, i64 %offset,
                                ptr addrspace(3) %src, ptr addrspace(3) %bar) {
; AA-LABEL: Function: put_generic_effects:
; AA-DAG: NoModRef:  Ptr: {{.*}}%generic{{[[:space:]]*}}<->{{[[:space:]]*}}{{.*}}@llvm.nvvm.fabric.handle_pair
; AA-DAG: Both ModRef:  Ptr: {{.*}}%generic{{[[:space:]]*}}<->{{[[:space:]]*}}call void @llvm.nvvm.fabric.try_put(
; GVN-LABEL: define i32 @put_generic_effects(
; GVN: %before = load i32, ptr %generic
; GVN: %after = load i32, ptr %generic
; GVN: %difference = sub i32 %after, %before
  %handle = call ptr addrspace(8) @llvm.nvvm.fabric.handle_pair(i32 %id, i64 %offset)
  %before = load i32, ptr %generic
  call void @llvm.nvvm.fabric.try_put(ptr addrspace(8) %handle, ptr addrspace(3) %src, ptr addrspace(3) %bar, i32 16,
                                      i16 0, i64 0, i1 false, i1 false, i32 0)
  %after = load i32, ptr %generic
  %difference = sub i32 %after, %before
  ret i32 %difference
}

; ATTR: declare ptr addrspace(8) @llvm.nvvm.fabric.handle_pair(i32, i64) #[[HANDLE:[0-9]+]]
; The put declarations are argument-memory-only: their resource accesses are modeled through the handle arguments.
; ATTR: declare void @llvm.nvvm.fabric.try_put(ptr addrspace(8), ptr addrspace(3) readonly, ptr addrspace(3), i32, i16, i64, i1 immarg, i1 immarg, i32 immarg range(i32 0, 2)) #[[PUT:[0-9]+]]
; ATTR: declare void @llvm.nvvm.fabric.try_put.counted_writes(ptr addrspace(8), ptr addrspace(8), ptr addrspace(3) readonly, ptr addrspace(3), i32, i64, i1 immarg, i32 immarg range(i32 0, 2)) #[[PUT]]
; ATTR: attributes #[[HANDLE]] = { {{.*}}memory(none){{.*}} }
; ATTR: attributes #[[PUT]] = { {{.*}}memory(argmem: readwrite){{.*}} }
