; RUN: mlir-translate -import-llvm -verify-diagnostics %s -o /dev/null
; RUN: mlir-translate -import-llvm -prefer-unregistered-intrinsics -verify-diagnostics %s -o /dev/null

declare void @llvm.assume(i1)
declare void @f()

define void @test() {
  ; expected-error @unknown {{unhandled constant: ptr no_cfi @f since no_cfi is unsupported}}
  ; expected-error @unknown {{unhandled intrinsic:}}
  call void @llvm.assume(i1 true) ["align"(ptr no_cfi @f, i64 16)]
  ret void
}
