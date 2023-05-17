; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Large vector for intrinsics is valid
; CHECK-NOT: llvm.fshr
define dso_local <8192 x i32> @test_intrin(<8192 x i32> %l, <8192 x i32> %r, <8192 x i32> %amt) {
entry:
  %b = call <8192 x i32> @llvm.fshr.v8192i32(<8192 x i32> %l, <8192 x i32> %r, <8192 x i32> %amt)
  ret <8192 x i32> %b
}
declare <8192 x i32> @llvm.fshr.v8192i32 (<8192 x i32> %l, <8192 x i32> %r, <8192 x i32> %amt)

; CHECK: Incorrect alignment of argument passed to called function!
; CHECK: bar
define dso_local void @foo(<8192 x float> noundef %vec) {
entry:
  call void @bar(<8192 x float> %vec)
  ret void
}

declare dso_local void @bar(<8192 x float>)
