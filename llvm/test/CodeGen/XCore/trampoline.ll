; RUN: llc < %s -march=xcore | FileCheck %s

%struct.FRAME.f = type { i32, ptr }

define void @f() nounwind {
entry:
; CHECK-LABEL: f:
; CHECK: ldap r11, g.1101
; CHECK: stw r11, sp[7]
  %TRAMP.23 = alloca [20 x i8], align 2
  %FRAME.0 = alloca %struct.FRAME.f, align 4
  call void @llvm.init.trampoline(ptr %TRAMP.23, ptr @g.1101, ptr %FRAME.0)
  %tramp = call ptr @llvm.adjust.trampoline(ptr %TRAMP.23)
  %0 = getelementptr inbounds %struct.FRAME.f, ptr %FRAME.0, i32 0, i32 1
  store ptr %tramp, ptr %0, align 4
  store i32 1, ptr %FRAME.0, align 4
  call void @h(ptr %tramp) nounwind
  ret void
}

define internal i32 @g.1101(ptr nocapture nest %CHAIN.1) nounwind readonly {
entry:
; CHECK: g.1101:
; CHECK: ldw r11, sp[0]
; CHECK-NEXT: ldw r0, r11[0]
; CHECK-NEXT: retsp 0
  %0 = load i32, ptr %CHAIN.1, align 4
  ret i32 %0
}

declare void @llvm.init.trampoline(ptr, ptr, ptr) nounwind
declare ptr @llvm.adjust.trampoline(ptr) nounwind

declare void @h(ptr)
