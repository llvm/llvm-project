; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @use(ptr)

; CHECK-LABEL: define ptr addrspace(3) @only_addrspace_alloca()
; CHECK:         %[[BASE:[0-9]+]] = phi ptr
; CHECK:         %[[SLOT:[0-9]+]] = getelementptr i8, ptr %[[BASE]], i64 32
; CHECK-NEXT:    %[[CAST:[0-9]+]] = addrspacecast ptr %[[SLOT]] to ptr addrspace(3)
; CHECK:         ret ptr addrspace(3) %[[CAST]]
define ptr addrspace(3) @only_addrspace_alloca() sanitize_address {
  %p = alloca float, align 4, addrspace(3)
  ret ptr addrspace(3) %p
}

; CHECK-LABEL: define ptr addrspace(3) @mixed_allocas()
; CHECK:         %[[BASE:[0-9]+]] = phi ptr
; CHECK:         %[[QSLOT:[0-9]+]] = getelementptr i8, ptr %[[BASE]], i64 32
; CHECK-NEXT:    %[[PSLOT:[0-9]+]] = getelementptr i8, ptr %[[BASE]], i64 48
; CHECK-NEXT:    %[[PCAST:[0-9]+]] = addrspacecast ptr %[[PSLOT]] to ptr addrspace(3)
; CHECK:         call void @use(ptr %[[QSLOT]])
; CHECK:         ret ptr addrspace(3) %[[PCAST]]
define ptr addrspace(3) @mixed_allocas() sanitize_address {
  %q = alloca i32, align 4
  %p = alloca float, align 4, addrspace(3)
  call void @use(ptr %q)
  ret ptr addrspace(3) %p
}
