; RUN: opt -S -passes=openmp-opt < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define ptx_kernel void @kernel_no_deinit
; CHECK:         %{{.*}} = call i32 @__kmpc_target_init(ptr null, ptr null)
define ptx_kernel void @kernel_no_deinit() {
entry:
  %0 = call i32 @__kmpc_target_init(ptr null, ptr null)
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit

user_code.entry:
  ret void

worker.exit:
  ret void
}

; CHECK-LABEL: define ptx_kernel void @kernel_with_deinit
; CHECK:         %{{.*}} = call i32 @__kmpc_target_init(ptr null, ptr null)
; CHECK:         call void @__kmpc_target_deinit()
define ptx_kernel void @kernel_with_deinit() #0 {
entry:
  %0 = call i32 @__kmpc_target_init(ptr null, ptr null)
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit

user_code.entry:
  call void @__kmpc_target_deinit()
  ret void

worker.exit:
  ret void
}

declare i32 @__kmpc_target_init(ptr, ptr)
declare void @__kmpc_target_deinit()

attributes #0 = { "kernel" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 50}
!1 = !{i32 7, !"openmp-device", i32 50}
