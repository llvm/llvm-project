; RUN: not llc -mtriple=powerpc64-unknown-linux -verify-machineinstrs 2>&1 < %s | FileCheck %s

define void @func(i64 %a, i64 %b) {
entry:
  %test = icmp slt i64 %a, %b
  br i1 %test, label %ret, label %cold
cold:
  %thunk = inttoptr i64 244837814094590 to ptr
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 4, i32 36, ptr %thunk, i32 0, i64 %a, i64 %b)
  unreachable
ret:
  ret void
}

; CHECK: LLVM ERROR: Patchpoint can't request size less than the length of a call.
