; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s

define void @main() {
  %res = call <4 x i32> @llvm.vp.sdiv.v4i32(<4 x i32> splat (i32 10), <4 x i32> <i32 2, i32 0, i32 0, i32 0>, <4 x i1> <i1 true, i1 true, i1 false, i1 false>, i32 2)
  ret void
}

; CHECK: Immediate UB detected: Division by zero.
