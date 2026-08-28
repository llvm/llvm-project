; RUN: not llc < %s --mtriple=wasm32-unknown-unknown -mattr=+reference-types 2>&1 | FileCheck %s

; We have only implemented a lowering for llvm.wasm.funcref.to_ptr its result
; feeds directly into an indirect call. Check that we diagnose the case where we
; spill the result rather than crashing in the backend.

%funcref = type target("wasm.funcref")

declare ptr @llvm.wasm.funcref.to_ptr(%funcref)
declare void @sink(ptr)

; CHECK: error: {{.*}}in function escape_via_store {{.*}}: a funcref can only be converted to a pointer to be directly called; the resulting pointer cannot otherwise be used
define void @escape_via_store(%funcref %ref, ptr %dst) {
  %p = call ptr @llvm.wasm.funcref.to_ptr(%funcref %ref)
  store ptr %p, ptr %dst
  ret void
}

; CHECK: error: {{.*}}in function escape_via_return {{.*}}: a funcref can only be converted to a pointer to be directly called; the resulting pointer cannot otherwise be used
define ptr @escape_via_return(%funcref %ref) {
  %p = call ptr @llvm.wasm.funcref.to_ptr(%funcref %ref)
  ret ptr %p
}

; CHECK: error: {{.*}}in function escape_via_arg {{.*}}: a funcref can only be converted to a pointer to be directly called; the resulting pointer cannot otherwise be used
define void @escape_via_arg(%funcref %ref) {
  %p = call ptr @llvm.wasm.funcref.to_ptr(%funcref %ref)
  call void @sink(ptr %p)
  ret void
}
