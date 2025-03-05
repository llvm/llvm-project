; RUN: opt < %s -passes=rewrite-statepoints-for-gc -S 2>&1 | FileCheck %s

define ptr addrspace(1) @test(ptr %a, ptr addrspace(1) %b) gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK-NEXT:    [[RES:%.*]] = atomicrmw xchg ptr %a, ptr addrspace(1) %b seq_cst
; CHECK-NEXT:    [[STATEPOINT_TOKEN:%.*]] = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) [ "gc-live"(ptr addrspace(1) [[RES]]) ]
; CHECK-NEXT:    [[RES_RELOCATED:%.*]] = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token [[STATEPOINT_TOKEN]], i32 0, i32 0)
; CHECK-NEXT:    ret ptr addrspace(1) [[RES_RELOCATED]]
  %res = atomicrmw xchg ptr %a, ptr addrspace(1) %b seq_cst
  call void @foo()
  ret ptr addrspace(1) %res
}

declare void @foo()
