; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=nehalem < %s | FileCheck %s

@a = common global [3 x i64] zeroinitializer, align 16

define i32 @main() nounwind ssp {
; CHECK: movups
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  call void @llvm.memset.p0.i64(ptr getelementptr inbounds ([3 x i64], ptr @a, i32 0, i64 1), i8 0, i64 16, i1 false)
  %0 = load i32, ptr %retval
  ret i32 %0
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind
