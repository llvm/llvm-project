; RUN: opt -objc-arc-contract -S < %s | FileCheck %s

declare i8* @llvm.objc.initWeak(i8**, i8*)

; Convert objc_initWeak(p, null) to *p = null.

; CHECK:      define i8* @test0(i8** %p) {
; CHECK-NEXT:   store i8* null, i8** %p
; CHECK-NEXT:   ret i8* null
; CHECK-NEXT: }
define i8* @test0(i8** %p) {
  %t = call i8* @llvm.objc.initWeak(i8** %p, i8* null)
  ret i8* %t
}
