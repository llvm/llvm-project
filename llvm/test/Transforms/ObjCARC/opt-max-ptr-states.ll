; RUN: opt -objc-arc -S < %s | FileCheck -check-prefix=ENABLE -check-prefix=CHECK %s
; RUN: opt -objc-arc -arc-opt-max-ptr-states=1 -S < %s | FileCheck -check-prefix=DISABLE -check-prefix=CHECK %s

@g0 = common global i8* null, align 8

; CHECK: call i8* @llvm.objc.retain
; ENABLE-NOT: call i8* @llvm.objc.retain
; DISABLE: call i8* @llvm.objc.retain
; CHECK: call void @llvm.objc.release
; ENABLE-NOT: call void @llvm.objc.release
; DISABLE: call void @llvm.objc.release

define void @foo0(i8* %a) {
  %1 = tail call i8* @llvm.objc.retain(i8* %a)
  %2 = tail call i8* @llvm.objc.retain(i8* %a)
  %3 = load i8*, i8** @g0, align 8
  store i8* %a, i8** @g0, align 8
  tail call void @llvm.objc.release(i8* %3)
  tail call void @llvm.objc.release(i8* %a), !clang.imprecise_release !0
  ret void
}

declare i8* @llvm.objc.retain(i8*)
declare void @llvm.objc.release(i8*)

!0 = !{}
