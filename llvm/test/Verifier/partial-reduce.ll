; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @element_count_mismatch() {
  ; CHECK: Invalid vector widths for partial reduction. The width of the input vector must be a known integer multiple of the width of the accumulator vector.
  call <3 x i32> @llvm.vector.partial.reduce.add(<3 x i32> poison, <8 x i32> poison)

  ; CHECK: Invalid vector widths for partial reduction. The width of the input vector must be a known integer multiple of the width of the accumulator vector.
  call <vscale x 4 x i32> @llvm.vector.partial.reduce.add(<vscale x 4 x i32> poison, <8 x i32> poison)

  ; CHECK: Invalid vector widths for partial reduction. The width of the input vector must be a known integer multiple of the width of the accumulator vector.
  call <4 x i32> @llvm.vector.partial.reduce.add(<4 x i32> poison, <vscale x 8 x i32> poison)
  ret void
}

define void @element_type_mismatch() {
  ; CHECK: Elements type of accumulator and input type must match
  call <4 x i32> @llvm.vector.partial.reduce.add(<4 x i32> poison, <8 x i8> poison)
  ret void
}
