; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@G = global [4 x i32] zeroinitializer

; CHECK-LABEL: @foo
; CHECK: ret <4 x ptr> getelementptr ([4 x i32], ptr @G, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>)
define <4 x ptr> @foo() {
  ret <4 x ptr> getelementptr ([4 x i32], ptr @G, i32 0, <4 x i32> <i32 0, i32 1, i32 2, i32 3>)
}
