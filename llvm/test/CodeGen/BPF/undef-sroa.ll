; RUN: opt -S -passes='sroa' %s -o %t1
; RUN: opt --bpf-check-undef-ir -S -mtriple=bpf-pc-linux %t1 >& %t2
; RUN: cat %t2 | FileCheck -check-prefixes=CHECK %s

define dso_local i32 @foo() {
  %1 = alloca [2 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 8, ptr %1)
  %2 = getelementptr inbounds [2 x i32], ptr %1, i64 0, i64 1
  %3 = load i32, ptr %2, align 4
  call void @llvm.lifetime.end.p0(i64 8, ptr %1)
  ret i32 %3
}
; CHECK: return undefined value in func foo, due to uninitialized variable?
; CHECK: ret i32 undef
