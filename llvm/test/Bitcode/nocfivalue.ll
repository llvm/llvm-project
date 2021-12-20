; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: @a = global [4 x void ()*] [void ()* no_cfi @f1, void ()* @f1, void ()* @f2, void ()* no_cfi @f2]
@a = global [4 x void ()*] [void ()* no_cfi @f1, void ()* @f1, void ()* @f2, void ()* no_cfi @f2]
; CHECK: @b = constant void ()* no_cfi @f3
@b = constant void ()* no_cfi @f3
; CHECK: @c = constant void ()* @f3
@c = constant void ()* @f3

; CHECK: declare void @f1()
declare void @f1()

; CHECK: declare void @f2()
declare void @f2()

; CHECK: define void @f3()
define void @f3() {
  ; CHECK: call void no_cfi @f4()
  call void no_cfi @f4()
  ; CHECK: call void @f4()
  call void @f4()
  ; CHECK: call void no_cfi @f5()
  call void no_cfi @f5()
  ; CHECK: call void @f5()
  call void @f5()
  ret void
}

; CHECK: declare void @f4()
declare void @f4()

; CHECK: declare void @f5()
declare void @f5()

define void @g() {
  %n = alloca void ()*, align 8
  ; CHECK: store void ()* no_cfi @f5, void ()** %n, align 8
  store void ()* no_cfi @f5, void ()** %n, align 8
  %1 = load void ()*, void ()** %n
  call void %1()
  ret void
}
