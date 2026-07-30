; RUN: llc < %s -mtriple x86_64-apple-darwin11 -O0 -fast-isel-abort=1 | FileCheck %s

%struct.x = type { i64, i64 }
declare %struct.x @f()

define void @test1(ptr) nounwind ssp {
  %2 = tail call %struct.x @f() nounwind
  %3 = extractvalue %struct.x %2, 0
  %4 = add i64 %3, 10
  store i64 %4, ptr %0
  ret void
; CHECK-LABEL: test1:
; CHECK: callq _f
; CHECK-NOT: %rax
; CHECK: addq $10, %rax
}

define void @test2(ptr) nounwind ssp {
  %2 = tail call %struct.x @f() nounwind
  %3 = extractvalue %struct.x %2, 1
  %4 = add i64 %3, 10
  store i64 %4, ptr %0
  ret void
; CHECK-LABEL: test2:
; CHECK: callq _f
; CHECK-NOT: %rdx
; CHECK: addq $10, %rdx
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone

define void @test3(i32 %x, i32 %y, ptr %z) {
  %r = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)
  %sum = extractvalue { i32, i1 } %r, 0
  %sum3 = mul i32 %sum, 3
  %bit = extractvalue { i32, i1 } %r, 1
  br i1 %bit, label %then, label %end
  
then:
  store i32 %sum3, ptr %z
  br label %end

end:
  ret void
; CHECK: test3
; CHECK: addl
; CHECK: seto %al
; CHECK: testb $1, %al
}
