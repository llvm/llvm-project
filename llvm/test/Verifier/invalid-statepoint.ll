; RUN: not opt -passes=verify 2>&1 < %s | FileCheck %s

declare zeroext i1 @return0i1()

; Function Attrs: nounwind
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...) #0

; Function Attrs: nounwind
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32) #0

; CHECK: gc.statepoint callee argument must have elementtype attribute
define ptr addrspace(1) @missing_elementtype(ptr addrspace(1) %dparam) gc "statepoint-example" {
  %a00 = load i32, ptr addrspace(1) %dparam
  %to0 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr @return0i1, i32 9, i32 0, i2 0) ["gc-live" (ptr addrspace(1) %dparam)]
  %relocate = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %to0, i32 0, i32 0)
  ret ptr addrspace(1) %relocate
}

; CHECK: gc.statepoint mismatch in number of call args
define ptr addrspace(1) @num_args_mismatch(ptr addrspace(1) %dparam) gc "statepoint-example" {
  %a00 = load i32, ptr addrspace(1) %dparam
  %to0 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return0i1, i32 9, i32 0, i2 0) ["gc-live" (ptr addrspace(1) %dparam)]
  %relocate = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %to0, i32 0, i32 0)
  ret ptr addrspace(1) %relocate
}

attributes #0 = { nounwind }
