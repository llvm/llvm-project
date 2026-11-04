; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; CHECK:   %v = call <4 x i32> @llvm.x86.sse2.pshuflw.128(<4 x i32> zeroinitializer, i8 0)
; CHECK: LLVM ERROR: Unexpected intrinsic

define void @test(ptr %a) {
  %v = call <4 x i32> @llvm.x86.sse2.pshuflw.128(<4 x i32> zeroinitializer, i8 0)
  store <4 x i32> %v, ptr %a
  ret void
}
