; RUN: opt %s -passes=bdce -S | FileCheck %s

; CHECK:      Function Attrs: convergent
; CHECK-NEXT: define i32 @foo(i32 %a) #0 {
define i32 @foo(i32 %a) #0 {
entry:
; CHECK-NOT: %tk0 = call token @llvm.experimental.convergence.entry()
  %tk0 = call token @llvm.experimental.convergence.entry()
  ret i32 %a
}

; CHECK:      Function Attrs: convergent
; CHECK-NEXT: define void @bar() #0 {
define void @bar() #0 {
entry:
; CHECK-NOT: %tk0 = call token @llvm.experimental.convergence.entry()
  %tk0 = call token @llvm.experimental.convergence.anchor()
  ret void
}

; CHECK:      Function Attrs: convergent
; CHECK-NEXT: define void @baz() #0 {
define void @baz() #0 {
entry:
; CHECK-NOT: %tk0 = call token @llvm.experimental.convergence.entry()
  %tk0 = call token @llvm.experimental.convergence.entry()
  br label %header

header:
; CHECK-NOT: %tk1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %tk0) ]
  %tk1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %tk0) ]
  br i1 true, label %body, label %exit

body:
  br label %header

exit:
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.anchor() #1
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

