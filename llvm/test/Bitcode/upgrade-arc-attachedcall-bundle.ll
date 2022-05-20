; Test that nullary clang.arc.attachedcall operand bundles are "upgraded".

; RUN: llvm-dis %s.bc -o - | FileCheck %s
; RUN: verify-uselistorder %s.bc

define i8* @invalid() {
; CHECK-LABEL: define i8* @invalid() {
; CHECK-NEXT:   %tmp0 = call i8* @foo(){{$}}
; CHECK-NEXT:   ret i8* %tmp0
  %tmp0 = call i8* @foo() [ "clang.arc.attachedcall"() ]
  ret i8* %tmp0
}

define i8* @valid() {
; CHECK-LABEL: define i8* @valid() {
; CHECK-NEXT:   %tmp0 = call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT:   ret i8* %tmp0
  %tmp0 = call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret i8* %tmp0
}

declare i8* @foo()
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
