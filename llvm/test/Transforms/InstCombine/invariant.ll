; Test to make sure unused llvm.invariant.start calls are not trivially eliminated
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare void @g(ptr)
declare void @g_addr1(ptr addrspace(1))

declare ptr @llvm.invariant.start.p0(i64, ptr nocapture) nounwind readonly
declare ptr @llvm.invariant.start.p1(i64, ptr addrspace(1) nocapture) nounwind readonly

define i8 @f() {
  %a = alloca i8                                  ; <ptr> [#uses=4]
  store i8 0, ptr %a
  %i = call ptr @llvm.invariant.start.p0(i64 1, ptr %a) ; <ptr> [#uses=0]
  ; CHECK: call ptr @llvm.invariant.start.p0
  call void @g(ptr %a)
  %r = load i8, ptr %a                                ; <i8> [#uses=1]
  ret i8 %r
}

; make sure llvm.invariant.call in non-default addrspace are also not eliminated.
define i8 @f_addrspace1(ptr addrspace(1) %a) {
  store i8 0, ptr addrspace(1) %a
  %i = call ptr @llvm.invariant.start.p1(i64 1, ptr addrspace(1) %a) ; <ptr> [#uses=0]
  ; CHECK: call ptr @llvm.invariant.start.p1
  call void @g_addr1(ptr addrspace(1) %a)
  %r = load i8, ptr addrspace(1) %a                                ; <i8> [#uses=1]
  ret i8 %r
}
