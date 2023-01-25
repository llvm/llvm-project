; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

declare ptr @llvm.objc.initWeak(ptr, ptr)

; Convert objc_initWeak(p, null) to *p = null.

; CHECK:      define ptr @test0(ptr %p) {
; CHECK-NEXT:   store ptr null, ptr %p
; CHECK-NEXT:   ret ptr null
; CHECK-NEXT: }
define ptr @test0(ptr %p) {
  %t = call ptr @llvm.objc.initWeak(ptr %p, ptr null)
  ret ptr %t
}
