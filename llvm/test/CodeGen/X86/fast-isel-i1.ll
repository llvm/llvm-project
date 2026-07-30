; RUN: llc < %s -mtriple=i686-apple-darwin10 -fast-isel -fast-isel-abort=1 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -fast-isel -fast-isel-abort=1 | FileCheck %s

declare i32 @test1a(i32)

define i32 @test1(i32 %x) nounwind {
; CHECK-LABEL: test1:
; CHECK: andb $1, %
	%y = add i32 %x, -3
	%t = call i32 @test1a(i32 %y)
	%s = mul i32 %t, 77
	%z = trunc i32 %s to i1
	br label %next

next:		; preds = %0
	%u = zext i1 %z to i32
	%v = add i32 %u, 1999
	br label %exit

exit:		; preds = %next
	ret i32 %v
}

define void @test2(ptr %a) nounwind {
entry:
; clang uses i8 constants for booleans, so we test with an i8 1.
; CHECK-LABEL: test2:
; CHECK: movzbl {{.*}} %eax
; CHECK-NEXT: xorb $1, %al
; CHECK-NEXT: testb $1
  %tmp = load i8, ptr %a, align 1
  %xor = xor i8 %tmp, 1
  %tobool = trunc i8 %xor to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:
  call void @test2(ptr null)
  br label %if.end

if.end:
  ret void
}
