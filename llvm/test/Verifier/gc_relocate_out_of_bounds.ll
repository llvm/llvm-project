; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

declare void @foo()

; CHECK: gc.relocate: statepoint base index out of bounds
define ptr addrspace(1) @test1(ptr addrspace(1) %a) gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %a)]
  %reloc = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token,  i32 1, i32 0)
  ret ptr addrspace(1) %reloc
}

; CHECK: gc.relocate: statepoint derived index out of bounds
define ptr addrspace(1) @test2(ptr addrspace(1) %a) gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %a)]
  %reloc = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token,  i32 0, i32 1)
  ret ptr addrspace(1) %reloc
}
