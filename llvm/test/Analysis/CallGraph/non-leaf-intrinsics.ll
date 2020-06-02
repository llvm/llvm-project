; RUN: opt -S -print-callgraph -disable-output < %s 2>&1 | FileCheck %s

declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

define private void @f() {
  ret void
}

define void @calls_statepoint(i8 addrspace(1)* %arg) gc "statepoint-example" {
entry:
  %cast = bitcast i8 addrspace(1)* %arg to i64 addrspace(1)*
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @f, i32 0, i32 0, i32 0, i32 0, i8 addrspace(1)* %arg, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg, i8 addrspace(1)* %arg) ["deopt" (i32 0, i32 0, i32 0, i32 10, i32 0)]
  ret void
}

define void @calls_patchpoint() {
entry:
  %c = bitcast void()* @f to i8*
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 15, i8* %c, i32 0, i16 65535, i16 -1, i32 65536, i32 2000000000, i32 2147483647, i32 -1, i32 4294967295, i32 4294967296, i64 2147483648, i64 4294967295, i64 4294967296, i64 -1)
  ret void
}


; CHECK: Call graph node <<null function>>
; CHECK:  CS<0x0> calls function 'f'

; CHECK: Call graph node for function: 'calls_patchpoint'
; CHECK-NEXT:  CS<[[addr_1:[^>]+]]> calls external node

; CHECK: Call graph node for function: 'calls_statepoint'
; CHECK-NEXT:  CS<[[addr_0:[^>]+]]> calls external node
