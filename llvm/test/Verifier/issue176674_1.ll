; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: Invalid Intrinsic Signature

define void @test(ptr %a) {
  %v = call <4 x i32> @llvm.x86.sse2.pshufl.w(<4 x i32> zeroinitializer, i8 0)
  store <4 x i32> %v, ptr %a
  ret void
}
