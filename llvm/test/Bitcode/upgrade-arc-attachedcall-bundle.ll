; Test that nullary clang.arc.attachedcall operand bundles are "upgraded".

; RUN: llvm-dis %s.bc -o - | FileCheck %s
; RUN: verify-uselistorder %s.bc

define i8* @invalid() {
; CHECK-LABEL: define ptr @invalid() {
; CHECK-NEXT:   %tmp0 = call ptr @foo(){{$}}
; CHECK-NEXT:   ret ptr %tmp0
  %tmp0 = call i8* @foo() [ "clang.arc.attachedcall"() ]
  ret i8* %tmp0
}

define i8* @valid() {
; CHECK-LABEL: define ptr @valid() {
; CHECK-NEXT:   %tmp0 = call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT:   ret ptr %tmp0
  %tmp0 = call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret i8* %tmp0
}

declare i8* @foo()
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
