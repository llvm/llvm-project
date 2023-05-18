; RUN: opt -S -O2 -inline-threshold=1 < %s | FileCheck %s

%class.A = type { i32 }

define void @_Z3barP1A(ptr %a) #0 {
entry:
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 10
  store i32 %add, ptr %a, align 4
  ret void
}

define void @_Z3foov() #0 {
; CHECK-LABEL: @_Z3foov(
; CHECK-NOT: call void @_Z3barP1A
; CHECK: ret
entry:
  %a = alloca %class.A, align 4
  call void @_Z3barP1A(ptr %a)
  ret void
}
