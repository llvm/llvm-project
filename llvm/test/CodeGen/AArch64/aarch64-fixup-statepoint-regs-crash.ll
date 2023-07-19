; XFAIL: *
; REQUIRES: asserts
; RUN: llc -verify-machineinstrs -max-registers-for-gc-values=256 -mtriple=aarch64-none-linux-gnu < %s

define dso_local ptr addrspace(1) @foo(ptr addrspace(1) %arg) gc "statepoint-example" personality ptr null {
  %load = load <2 x ptr addrspace(1)>, ptr addrspace(1) %arg, align 8
  %extractelement = extractelement <2 x ptr addrspace(1)> %load, i64 0
  %call = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr nonnull elementtype(void ()) @baz, i32 0, i32 0, i32 0, i32 0) [ "deopt"(ptr addrspace(1) %extractelement), "gc-live"(<2 x ptr addrspace(1)> %load) ]
  %relocate = call coldcc <2 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v2p1(token %call, i32 0, i32 0)
  %extractelement2 = extractelement <2 x ptr addrspace(1)> %relocate, i64 0
  ret ptr addrspace(1) %extractelement2
}

declare void @baz()
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <2 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v2p1(token, i32, i32) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
