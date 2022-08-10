; RUN: opt -S -disable-output -passes=print-lcg < %s 2>&1 | FileCheck %s

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, ptr, i32, i32, ...)

define private void @f() {
  ret void
}

define void @calls_statepoint(ptr addrspace(1) %arg) gc "statepoint-example" {
; CHECK: Edges in function: calls_statepoint
; CHECK-NEXT:  -> f
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, ptr elementtype(void ()) @f, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg), "deopt" (i32 0, i32 0, i32 0, i32 10, i32 0)]
  ret void
}

define void @calls_patchpoint() {
; CHECK:  Edges in function: calls_patchpoint
; CHECK-NEXT:    -> f
entry:
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 15, ptr @f, i32 0, i16 65535, i16 -1, i32 65536, i32 2000000000, i32 2147483647, i32 -1, i32 4294967295, i32 4294967296, i64 2147483648, i64 4294967295, i64 4294967296, i64 -1)
  ret void
}
