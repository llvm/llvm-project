; RUN: llvm-as < %s | opt -always-inline | llvm-dis | grep {@foo}
; Ensure that foo is not removed by always inliner
; PR 2945

define internal i32 @foo() nounwind {
  ret i32 0
}
