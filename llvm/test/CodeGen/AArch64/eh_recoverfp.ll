; RUN: llc -mtriple arm64-windows %s -o - 2>&1 | FileCheck %s

define i8* @foo(i8* %a) {
; CHECK-LABEL: foo
; CHECK-NOT: llvm.x86.seh.recoverfp
  %1 = call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @f to i8*), i8* %a)
  ret i8* %1
}

declare i8* @llvm.x86.seh.recoverfp(i8*, i8*)
declare i32 @f()
