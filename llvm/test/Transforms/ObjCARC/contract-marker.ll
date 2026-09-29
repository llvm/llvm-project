; RUN: opt -S -passes=objc-arc-contract < %s | FileCheck %s

; CHECK-LABEL: define void @foo() {
; CHECK:      %call = tail call ptr @qux()
; CHECK-NEXT: call void asm sideeffect "mov\09r7, r7\09\09@ marker for return value optimization", ""()
; CHECK-NEXT: %0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) [[NUW:#[0-9]+]]
; CHECK: }

define void @foo() {
entry:
  %call = tail call ptr @qux()
  %0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  tail call void @bar(ptr %0)
  ret void
}

; CHECK-LABEL: define void @foo2() {
; CHECK:      %call = tail call ptr @qux()
; CHECK-NEXT: call void asm sideeffect "mov\09r7, r7\09\09@ marker for return value optimization", ""()
; CHECK-NEXT: %0 = tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %call) [[NUW:#[0-9]+]]
; CHECK: }

define void @foo2() {
entry:
  %call = tail call ptr @qux()
  %0 = tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %call) nounwind
  tail call void @bar(ptr %0)
  ret void
}

; CHECK-LABEL: define ptr @foo3(
; CHECK: call ptr @returnsArg(
; CHECK-NEXT: call void asm sideeffect

define ptr @foo3(ptr %a) {
  %call = call ptr @returnsArg(ptr %a)
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call)
  ret ptr %call
}

; CHECK-LABEL: define ptr @foo4(
; CHECK: call ptr @returnsArg(
; CHECK-NEXT: call void asm sideeffect

define ptr @foo4(ptr %a) {
  %call = call ptr @returnsArg(ptr %a)
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %a)
  ret ptr %call
}

declare ptr @qux()
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
declare void @bar(ptr)
declare ptr @returnsArg(ptr returned)

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov\09r7, r7\09\09@ marker for return value optimization"}

; CHECK: attributes [[NUW]] = { nounwind }
