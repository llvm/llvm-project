; RUN: opt -mtriple=arm64e-apple-ios -passes=pre-isel-intrinsic-lowering -S -o - %s | FileCheck %s --check-prefix=ARM64E
; RUN: opt -mtriple=arm64-apple-ios -passes=pre-isel-intrinsic-lowering -S -o - %s | FileCheck %s --check-prefix=ARM64
;
; Test that objc_retain and objc_release do not get nonlazybind on arm64e.

define ptr @test_objc_retain(ptr %arg0) {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %arg0)
  ret ptr %0
}

define void @test_objc_release(ptr %arg0) {
entry:
  call void @llvm.objc.release(ptr %arg0)
  ret void
}

declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.retain(ptr)

; arm64e: retain and release should NOT have nonlazybind.
; ARM64E-DAG: declare void @objc_release(ptr)
; ARM64E-DAG: declare ptr @objc_retain(ptr)
; ARM64E-NOT: nonlazybind

; arm64: retain and release should have nonlazybind.
; ARM64-DAG: declare void @objc_release(ptr) [[NLB:#[0-9]+]]
; ARM64-DAG: declare ptr @objc_retain(ptr) [[NLB]]
; ARM64: attributes [[NLB]] = { nonlazybind }
