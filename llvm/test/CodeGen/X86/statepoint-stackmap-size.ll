; RUN: llc  -verify-machineinstrs < %s | FileCheck %s

; Without removal of duplicate entries, the size is 62 lines
;      CHECK:	.section	.llvm_stackmaps,{{.*$}}
; CHECK-NEXT:{{(.+$[[:space:]]){48}[[:space:]]}}
;  CHECK-NOT:{{.|[[:space:]]}}

target triple = "x86_64-pc-linux-gnu"

declare void @func()

define i1 @test1(ptr addrspace(1) %arg) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %arg)]
  %reloc1 = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token,  i32 0, i32 0)
  %reloc2 = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token,  i32 0, i32 0)
  %cmp1 = icmp eq ptr addrspace(1) %reloc1, null
  %cmp2 = icmp eq ptr addrspace(1) %reloc2, null
  %cmp = and i1 %cmp1, %cmp2
  ret i1 %cmp
}

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)
