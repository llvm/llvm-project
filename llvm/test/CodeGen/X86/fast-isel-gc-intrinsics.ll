; RUN: llc < %s -fast-isel

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
; Dont crash with gc intrinsics.

; gcrelocate call should not be an LLVM Machine Block by itself.
define ptr addrspace(1) @test_gcrelocate(ptr addrspace(1) %v) gc "statepoint-example" {
entry:
  %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
  %vnew = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok,  i32 0, i32 0)
  ret ptr addrspace(1) %vnew
}

; gcresult calls are fine in their own blocks.
define i1 @test_gcresult() gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0)
  %call1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  br label %exit
exit:
  ret i1 %call1
}

; we are okay here because we see the gcrelocate and avoid generating their own
; block.
define i1 @test_gcresult_gcrelocate(ptr addrspace(1) %v) gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
  %call1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  %vnew = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token,  i32 0, i32 0)
  br label %exit
exit:
  ret i1 %call1
}

define ptr addrspace(1)  @test_non_entry_block(ptr addrspace(1) %v, i8 %val) gc "statepoint-example" {
entry:
 %load = load i8, ptr addrspace(1) %v
 %cmp = icmp eq i8 %load, %val
 br i1 %cmp, label %func_call, label %exit

func_call:
 call void @dummy()
 %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
 %vnew = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok,  i32 0, i32 0)
 ret ptr addrspace(1) %vnew

exit:
  ret ptr addrspace(1) %v

}

declare void @dummy()
declare void @foo()

declare zeroext i1 @return_i1()
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare i1 @llvm.experimental.gc.result.i1(token)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)
