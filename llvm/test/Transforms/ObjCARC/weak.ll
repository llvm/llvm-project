; RUN: opt -objc-arc -S < %s | FileCheck %s

declare i8* @llvm.objc.initWeak(i8**, i8*)
declare i8* @llvm.objc.storeWeak(i8**, i8*)
declare i8* @llvm.objc.loadWeak(i8**)
declare void @llvm.objc.destroyWeak(i8**)
declare i8* @llvm.objc.loadWeakRetained(i8**)
declare void @llvm.objc.moveWeak(i8**, i8**)
declare void @llvm.objc.copyWeak(i8**, i8**)

; If the pointer-to-weak-pointer is null, it's undefined behavior.

; CHECK-LABEL: define void @test0(
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: store i8* undef, i8** null
; CHECK: ret void
define void @test0(i8* %p, i8** %q) {
entry:
  call i8* @llvm.objc.storeWeak(i8** null, i8* %p)
  call i8* @llvm.objc.storeWeak(i8** undef, i8* %p)
  call i8* @llvm.objc.loadWeakRetained(i8** null)
  call i8* @llvm.objc.loadWeakRetained(i8** undef)
  call i8* @llvm.objc.loadWeak(i8** null)
  call i8* @llvm.objc.loadWeak(i8** undef)
  call i8* @llvm.objc.initWeak(i8** null, i8* %p)
  call i8* @llvm.objc.initWeak(i8** undef, i8* %p)
  call void @llvm.objc.destroyWeak(i8** null)
  call void @llvm.objc.destroyWeak(i8** undef)

  call void @llvm.objc.copyWeak(i8** null, i8** %q)
  call void @llvm.objc.copyWeak(i8** undef, i8** %q)
  call void @llvm.objc.copyWeak(i8** %q, i8** null)
  call void @llvm.objc.copyWeak(i8** %q, i8** undef)

  call void @llvm.objc.moveWeak(i8** null, i8** %q)
  call void @llvm.objc.moveWeak(i8** undef, i8** %q)
  call void @llvm.objc.moveWeak(i8** %q, i8** null)
  call void @llvm.objc.moveWeak(i8** %q, i8** undef)

  ret void
}
