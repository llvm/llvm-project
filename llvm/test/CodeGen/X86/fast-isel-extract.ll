; RUN: llc < %s -mtriple x86_64-apple-darwin11 -O0 -fast-isel-abort=1 | FileCheck %s

%struct.x = type { i64, i64 }
%addovf = type { i32, i1 }
declare %struct.x @f()

define void @test1(i64*) nounwind ssp {
  %2 = tail call %struct.x @f() nounwind
  %3 = extractvalue %struct.x %2, 0
  %4 = add i64 %3, 10
  store i64 %4, i64* %0
  ret void
; CHECK-LABEL: test1:
; CHECK: callq _f
; CHECK-NOT: %rax
; CHECK: addq $10, %rax
}

define void @test2(i64*) nounwind ssp {
  %2 = tail call %struct.x @f() nounwind
  %3 = extractvalue %struct.x %2, 1
  %4 = add i64 %3, 10
  store i64 %4, i64* %0
  ret void
; CHECK-LABEL: test2:
; CHECK: callq _f
; CHECK-NOT: %rdx
; CHECK: addq $10, %rdx
}

declare %addovf @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone

define void @test3(i32 %x, i32 %y, i32* %z) {
  %r = call %addovf @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)
  %sum = extractvalue %addovf %r, 0
  %sum3 = mul i32 %sum, 3
  %bit = extractvalue %addovf %r, 1
  br i1 %bit, label %then, label %end
  
then:
  store i32 %sum3, i32* %z
  br label %end

end:
  ret void
; CHECK: test3
; CHECK: addl
; CHECK: seto %al
; CHECK: testb $1, %al
}
