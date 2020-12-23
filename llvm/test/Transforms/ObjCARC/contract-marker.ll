; RUN: opt -S -objc-arc-contract < %s | FileCheck %s

; CHECK-LABEL: define void @foo() {
; CHECK:      %call = tail call i32* @qux()
; CHECK-NEXT: %tcall = bitcast i32* %call to i8*
; CHECK-NEXT: call void asm sideeffect "mov\09r7, r7\09\09@ marker for return value optimization", ""()
; CHECK-NEXT: %0 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %tcall) [[NUW:#[0-9]+]]
; CHECK: }

define void @foo() {
entry:
  %call = tail call i32* @qux()
  %tcall = bitcast i32* %call to i8*
  %0 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %tcall) nounwind
  tail call void @bar(i8* %0)
  ret void
}

; CHECK-LABEL: define void @foo2() {
; CHECK:      %call = tail call i32* @qux()
; CHECK-NEXT: %tcall = bitcast i32* %call to i8*
; CHECK-NEXT: call void asm sideeffect "mov\09r7, r7\09\09@ marker for return value optimization", ""()
; CHECK-NEXT: %0 = tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %tcall) [[NUW:#[0-9]+]]
; CHECK: }

define void @foo2() {
entry:
  %call = tail call i32* @qux()
  %tcall = bitcast i32* %call to i8*
  %0 = tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %tcall) nounwind
  tail call void @bar(i8* %0)
  ret void
}

; CHECK-LABEL: define i8* @foo3(
; CHECK: call i8* @returnsArg(
; CHECK-NEXT: call void asm sideeffect

define i8* @foo3(i8* %a) {
  %call = call i8* @returnsArg(i8* %a)
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call)
  ret i8* %call
}

; CHECK-LABEL: define i8* @foo4(
; CHECK: call i8* @returnsArg(
; CHECK-NEXT: call void asm sideeffect

define i8* @foo4(i8* %a) {
  %call = call i8* @returnsArg(i8* %a)
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %a)
  ret i8* %call
}

declare i32* @qux()
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)
declare void @bar(i8*)
declare i8* @returnsArg(i8* returned)

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov\09r7, r7\09\09@ marker for return value optimization"}

; CHECK: attributes [[NUW]] = { nounwind }
