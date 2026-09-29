; RUN: not --crash llc -mtriple arm64-apple-ios7 -mattr=+sve < %s 2>&1 | FileCheck %s

; CHECK: Passing SVE types to variadic functions is currently not supported

@.str = private unnamed_addr constant [4 x i8] c"fmt\00", align 1
define void @foo(ptr %fmt, ...) nounwind {
entry:
  %fmt.addr = alloca ptr, align 8
  %args = alloca ptr, align 8
  %vc = alloca i32, align 4
  %vv = alloca <vscale x 4 x i32>, align 16
  store ptr %fmt, ptr %fmt.addr, align 8
  call void @llvm.va_start(ptr %args)
  %0 = va_arg ptr %args, i32
  store i32 %0, ptr %vc, align 4
  %1 = va_arg ptr %args, <vscale x 4 x i32>
  store <vscale x 4 x i32> %1, ptr %vv, align 16
  ret void
}

declare void @llvm.va_start(ptr) nounwind
