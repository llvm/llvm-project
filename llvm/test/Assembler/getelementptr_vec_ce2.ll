; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@G = global [4 x [4 x i32]] zeroinitializer

; CHECK: getelementptr vector index has a wrong number of elements
define <4 x i32*> @foo() {
  ret <4 x i32*> getelementptr ([4 x [4 x i32]], [4 x [4 x i32]]* @G, i32 0, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <8 x i32> zeroinitializer)
}
