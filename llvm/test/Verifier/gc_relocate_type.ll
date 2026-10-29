; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s
; This is to verify that gc_relocate must return the type of the relocated
; value.

declare void @foo()

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)

; CHECK:       gc.relocate: must return the type of the relocated value
; CHECK-NEXT:  %obj.relocated = call coldcc ptr @llvm.experimental.gc.relocate.p0(token %safepoint_token, i32 0, i32 0)
define void @addrspace(ptr addrspace(1) %obj) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) [ "gc-live"(ptr addrspace(1) %obj) ]
  %obj.relocated = call coldcc ptr @llvm.experimental.gc.relocate.p0(token %safepoint_token, i32 0, i32 0)
  ret void
}

; CHECK:       gc.relocate: must return the type of the relocated value
; CHECK-NEXT:  %obj.relocated = call coldcc <2 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v2p1(token %safepoint_token, i32 0, i32 0)
define void @vector_width(<4 x ptr addrspace(1)> %obj) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) [ "gc-live"(<4 x ptr addrspace(1)> %obj) ]
  %obj.relocated = call coldcc <2 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v2p1(token %safepoint_token, i32 0, i32 0)
  ret void
}

; CHECK:       gc.relocate: must return the type of the relocated value
; CHECK-NEXT:  %obj.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 0)
define void @vector_to_scalar(<2 x ptr addrspace(1)> %obj) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) [ "gc-live"(<2 x ptr addrspace(1)> %obj) ]
  %obj.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 0)
  ret void
}
