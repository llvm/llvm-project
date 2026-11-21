; RUN: mlir-translate -import-llvm -prefer-unregistered-intrinsics -verify-diagnostics %s | FileCheck %s

declare void @llvm.assume(i1)
declare void @f()

; CHECK-LABEL: llvm.func @test(
; CHECK-SAME: %[[PTR:.*]]: !llvm.ptr)
; CHECK: llvm.call_intrinsic "llvm.assume"({{.*}}) ["nonnull"(%[[PTR]] : !llvm.ptr)]
define void @test(ptr %p) {
  ; expected-warning @unknown {{unhandled constant: ptr no_cfi @f since no_cfi is unsupported}}
  ; expected-warning @unknown {{dropping operand bundle 'align'}}
  call void @llvm.assume(i1 true) ["align"(ptr no_cfi @f, i64 16), "nonnull"(ptr %p)]
  ret void
}
