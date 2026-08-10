; RUN: llc -mtriple=armv7 < %s
; PR15053

declare i32 @llvm.arm.strexd(i32, i32, ptr) nounwind
declare { i32, i32 } @llvm.arm.ldrexd(ptr) nounwind readonly

define void @foo() {
entry:
  %0 = tail call { i32, i32 } @llvm.arm.ldrexd(ptr undef) nounwind
  %1 = extractvalue { i32, i32 } %0, 0
  %2 = tail call i32 @llvm.arm.strexd(i32 %1, i32 undef, ptr undef) nounwind
  ret void
}
