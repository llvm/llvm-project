; RUN: llc < %s -march=x86-64 -O0 | FileCheck %s
; Make sure fast-isel doesn't reset the materialised constant map
; across an intrinsic call.

; CHECK: movl	$100000
; CHECK-NOT: movl	$100000
define i1 @test1(i32 %v1, i32 %v2, i32* %X) nounwind {
entry:
  %a = shl i32 100000, %v1
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %v2)
  %ext = extractvalue {i32, i1} %t, 0
  %sum = shl i32 100000, %ext
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  store i32 %sum, i32* %X
  br label %overflow

overflow:
  ret i1 false
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
