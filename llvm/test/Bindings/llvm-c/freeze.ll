; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo --opaque-pointers > %t.echo
; RUN: diff -w %t.orig %t.echo

%struct.T = type { i32, i32 }

define i32 @f(i32 %arg, <2 x i32> %arg2, float %arg3, <2 x float> %arg4,
              ptr %arg5, %struct.T %arg6, [2 x i32] %arg7, { i32, i32 } %arg8) {
  %1 = freeze i32 %arg
  %2 = freeze i32 10
  %3 = freeze i32 %1
  %4 = freeze i32 undef
  %5 = freeze i666 11
  %6 = freeze <2 x i32> %arg2
  %7 = freeze float %arg3
  %8 = freeze <2 x float> %arg4
  %9 = freeze ptr %arg5
  %10 = freeze %struct.T %arg6
  %11 = freeze [2 x i32] %arg7
  %12 = freeze { i32, i32 } %arg8
  %13 = freeze ptr null
  ret i32 %1
}
