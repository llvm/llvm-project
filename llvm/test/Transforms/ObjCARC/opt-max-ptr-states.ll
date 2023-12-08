; RUN: opt -passes=objc-arc -S < %s | FileCheck -check-prefix=ENABLE -check-prefix=CHECK %s
; RUN: opt -passes=objc-arc -arc-opt-max-ptr-states=1 -S < %s | FileCheck -check-prefix=DISABLE -check-prefix=CHECK %s

@g0 = common global ptr null, align 8

; CHECK: call ptr @llvm.objc.retain
; ENABLE-NOT: call ptr @llvm.objc.retain
; DISABLE: call ptr @llvm.objc.retain
; CHECK: call void @llvm.objc.release
; ENABLE-NOT: call void @llvm.objc.release
; DISABLE: call void @llvm.objc.release

define void @foo0(ptr %a) {
  %1 = tail call ptr @llvm.objc.retain(ptr %a)
  %2 = tail call ptr @llvm.objc.retain(ptr %a)
  %3 = load ptr, ptr @g0, align 8
  store ptr %a, ptr @g0, align 8
  tail call void @llvm.objc.release(ptr %3)
  tail call void @llvm.objc.release(ptr %a), !clang.imprecise_release !0
  ret void
}

declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.release(ptr)

!0 = !{}
