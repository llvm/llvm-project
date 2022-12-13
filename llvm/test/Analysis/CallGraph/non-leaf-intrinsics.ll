; RUN: opt -S -passes=print-callgraph -disable-output < %s 2>&1 | FileCheck %s

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)

define private void @f() {
  ret void
}

define void @calls_statepoint(ptr addrspace(1) %arg) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...)
  @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @f, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg), "deopt" (i32 0, i32 0, i32 0, i32 10, i32 0)]
  ret void
}

define void @calls_patchpoint() {
entry:
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 15, ptr @f, i32 0, i16 65535, i16 -1, i32 65536, i32 2000000000, i32 2147483647, i32 -1, i32 4294967295, i32 4294967296, i64 2147483648, i64 4294967295, i64 4294967296, i64 -1)
  ret void
}


; CHECK: Call graph node <<null function>>
; CHECK:  CS<None> calls function 'f'

; CHECK: Call graph node for function: 'calls_patchpoint'
; CHECK-NEXT:  CS<[[addr_1:[^>]+]]> calls external node

; CHECK: Call graph node for function: 'calls_statepoint'
; CHECK-NEXT:  CS<[[addr_0:[^>]+]]> calls external node
