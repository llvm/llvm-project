; RUN: llc -o - %s | FileCheck %s

; Verify that ObjCARCContract rewrites retainRV to claimRV, removing the
; retainRV marker.

target triple = "arm64-apple-ios18"

declare ptr @f()

; CHECK-LABEL: _t:
; CHECK:         bl _f
; CHECK-NEXT:    bl _objc_claimAutoreleasedReturnValue
; CHECK-NOT:     mov x29, x29
define ptr @t() {
  %call = call ptr @f() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret ptr %call
}

declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
