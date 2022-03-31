; Test to make sure unused llvm.invariant.start calls are not trivially eliminated
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare void @g(i8*)
declare void @g_addr1(i8 addrspace(1)*)

declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly
declare {}* @llvm.invariant.start.p1i8(i64, i8 addrspace(1)* nocapture) nounwind readonly

define i8 @f() {
  %a = alloca i8                                  ; <i8*> [#uses=4]
  store i8 0, i8* %a
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %a) ; <{}*> [#uses=0]
  ; CHECK: call {}* @llvm.invariant.start.p0i8
  call void @g(i8* %a)
  %r = load i8, i8* %a                                ; <i8> [#uses=1]
  ret i8 %r
}

; make sure llvm.invariant.call in non-default addrspace are also not eliminated.
define i8 @f_addrspace1(i8 addrspace(1)* %a) {
  store i8 0, i8 addrspace(1)* %a
  %i = call {}* @llvm.invariant.start.p1i8(i64 1, i8 addrspace(1)* %a) ; <{}*> [#uses=0]
  ; CHECK: call {}* @llvm.invariant.start.p1i8
  call void @g_addr1(i8 addrspace(1)* %a)
  %r = load i8, i8 addrspace(1)* %a                                ; <i8> [#uses=1]
  ret i8 %r
}
