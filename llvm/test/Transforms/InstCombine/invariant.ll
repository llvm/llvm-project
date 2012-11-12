; Test to make sure unused llvm.invariant.start calls are not trivially eliminated
; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @g(i8*)

declare {}* @llvm.invariant.start(i64, i8* nocapture) nounwind readonly

define i8 @f() {
  %a = alloca i8                                  ; <i8*> [#uses=4]
  store i8 0, i8* %a
  %i = call {}* @llvm.invariant.start(i64 1, i8* %a) ; <{}*> [#uses=0]
  ; CHECK: call {}* @llvm.invariant.start
  call void @g(i8* %a)
  %r = load i8* %a                                ; <i8> [#uses=1]
  ret i8 %r
}
