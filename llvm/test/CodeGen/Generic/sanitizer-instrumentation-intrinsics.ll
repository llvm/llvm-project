; RUN: llc -O0 < %s | FileCheck %s
; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O0 -fast-isel < %s | FileCheck %s

; Verify that the backend lowers the sanitizer region intrinsics to nothing and
; does not crash when compiling a module where no sanitizer pass ran, on
; whatever the default target is. (The exact output is checked for X86 in
; test/CodeGen/X86/sanitizer-instrumentation-intrinsics.ll.)

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()

define void @region_markers() {
; CHECK-LABEL: region_markers:
; CHECK-NOT:     instrumentation
; CHECK-NOT:     call
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}
