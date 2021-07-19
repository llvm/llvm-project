; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

define protected amdgpu_kernel void @global_store(i32 addrspace(1)* %p, i32 %i) sanitize_address {
entry:
; CHECK-LABEL: @global_store
; CHECK-NOT: store
;
; CHECK:   %[[STORE_ADDR:[^ ]*]] = ptrtoint i32 addrspace(1)* %p to i64
; CHECK:   lshr i64 %[[STORE_ADDR]], 3
; CHECK:   add i64 %{{.*}}, 2147450880
; CHECK:   %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[STORE_SHADOW:[^ ]*]] = load i8, i8* %[[STORE_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; CHECK:   add i64 %{{.*}}, 3
; CHECK:   trunc i64 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[STORE_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_store4(i64 %[[STORE_ADDR]])
; CHECK:   unreachable
;
; The actual store.
; CHECK:   store i32 0, i32 addrspace(1)* %p
; CHECK:   ret void

  store i32 0, i32 addrspace(1)* %p, align 4
  ret void
}

define protected amdgpu_kernel void @global_load(i32 addrspace(1)* %p, i32 %i) sanitize_address {
entry:
; CHECK-LABEL: @global_load
; CHECK-NOT: load
;
; CHECK:   %[[LOAD_ADDR:[^ ]*]] = ptrtoint i32 addrspace(1)* %p to i64
; CHECK:   lshr i64 %[[LOAD_ADDR]], 3
; CHECK:   add i64 %{{.*}}, 2147450880
; CHECK:   %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK:   %[[LOAD_SHADOW:[^ ]*]] = load i8, i8* %[[LOAD_SHADOW_PTR]]
; CHECK:   icmp ne i8
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; CHECK:   add i64 %{{.*}}, 3
; CHECK:   trunc i64 %{{.*}} to i8
; CHECK:   icmp sge i8 %{{.*}}, %[[LOAD_SHADOW]]
; CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
;
; The crash block reports the error.
; CHECK:   call void @__asan_report_load4(i64 %[[LOAD_ADDR]])
; CHECK:   unreachable
;
; The actual load.
; CHECK:   load i32, i32 addrspace(1)* %p
; CHECK:   ret void

  %q = load i32, i32 addrspace(1)* %p, align 4
  ret void
}
