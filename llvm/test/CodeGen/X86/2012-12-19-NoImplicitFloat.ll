; RUN: llc -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core2 < %s | FileCheck %s
; Test that we do not introduce vector operations with noimplicitfloat.
; rdar://12879313

%struct1 = type { ptr, ptr }

define void @test() nounwind noimplicitfloat {
entry:
; CHECK-NOT: xmm
; CHECK: ret
  %0 = load ptr, ptr undef, align 8
  store ptr null, ptr %0, align 8
  %1 = getelementptr inbounds %struct1, ptr %0, i64 0, i32 1
  store ptr null, ptr %1, align 8
  ret void
}
