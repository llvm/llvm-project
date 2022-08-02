; RUN: opt %s -early-cse -S | FileCheck %s

define void @fn() {
entry:
  %0 = tail call ptr @llvm.dx.create.handle(i8 1)
  %1 = tail call ptr @llvm.dx.create.handle(i8 1)
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare ptr @llvm.dx.create.handle(i8) #0

attributes #0 = { mustprogress nounwind willreturn }

; CSE needs to leave this alone
; CHECK: %0 = tail call ptr @llvm.dx.create.handle(i8 1)
; CHECK: %1 = tail call ptr @llvm.dx.create.handle(i8 1)