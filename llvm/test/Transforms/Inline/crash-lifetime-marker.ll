; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; RUN: opt < %s -passes='module-inline' -S | FileCheck %s

; InlineFunction would assert inside the loop that leaves lifetime markers if
; there was an zero-sized AllocaInst. Check that it doesn't assert and doesn't
; leave lifetime markers in that case.

declare i32 @callee2(i8*)

define i32 @callee1(i32 %count) {
  %a0 = alloca i8, i32 %count, align 4
  %call0 = call i32 @callee2(i8* %a0)
  ret i32 %call0
}

; CHECK-LABEL: define i32 @caller1(
; CHECK: [[ALLOCA:%[a-z0-9\.]+]] = alloca i8
; CHECK-NOT: call void @llvm.lifetime.start.p0i8(
; CHECK: call i32 @callee2(i8* [[ALLOCA]])
; CHECK-NOT: call void @llvm.lifetime.end.p0i8(

define i32 @caller1(i32 %count) {
  %call0 = call i32 @callee1(i32 0)
  ret i32 %call0
}
