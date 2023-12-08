; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 | FileCheck %s

@0 = private constant [8 x i32] zeroinitializer

; CHECK-LABEL: foo:
; CHECK: movl  %esi, (%rdi)
; CHECK-NEXT: retq
define void @foo(ptr %p, i32 %v, <8 x i1> %mask) {
  store i32 %v, ptr %p
  %wide.masked.load = call <8 x i32> @llvm.masked.load.v8i32.p0(ptr @0, i32 4, <8 x i1> %mask, <8 x i32> undef)  
  ret void
}

declare <8 x i32> @llvm.masked.load.v8i32.p0(ptr, i32, <8 x i1>, <8 x i32>) #0

attributes #0 = { argmemonly nounwind readonly }
