; RUN: not --crash llc -mtriple arm64-apple-ios7 -mattr=+sve < %s 2>&1 | FileCheck %s

; CHECK: Passing SVE types to variadic functions is currently not supported

@.str = private unnamed_addr constant [4 x i8] c"fmt\00", align 1
define void @foo(i8* %fmt, ...) nounwind {
entry:
  %fmt.addr = alloca i8*, align 8
  %args = alloca i8*, align 8
  %vc = alloca i32, align 4
  %vv = alloca <vscale x 4 x i32>, align 16
  store i8* %fmt, i8** %fmt.addr, align 8
  %args1 = bitcast i8** %args to i8*
  call void @llvm.va_start(i8* %args1)
  %0 = va_arg i8** %args, i32
  store i32 %0, i32* %vc, align 4
  %1 = va_arg i8** %args, <vscale x 4 x i32>
  store <vscale x 4 x i32> %1, <vscale x 4 x i32>* %vv, align 16
  ret void
}

declare void @llvm.va_start(i8*) nounwind
