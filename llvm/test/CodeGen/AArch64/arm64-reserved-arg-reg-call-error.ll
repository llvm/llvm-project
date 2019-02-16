; RUN: not llc < %s -mtriple=arm64-linux-gnu -mattr=+reserve-x1 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=arm64-linux-gnu -mattr=+reserve-x1 -fast-isel 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=arm64-linux-gnu -mattr=+reserve-x1 -global-isel 2>&1 | FileCheck %s

; CHECK: error:
; CHECK-SAME: AArch64 doesn't support function calls if any of the argument registers is reserved.
define void @call_function() {
  call void @foo()
  ret void
}
declare void @foo()

; CHECK: error:
; CHECK-SAME: AArch64 doesn't support function calls if any of the argument registers is reserved.
define void @call_memcpy(i8* %out, i8* %in) {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %out, i8* %in, i64 800, i1 false)
  ret void
}
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
