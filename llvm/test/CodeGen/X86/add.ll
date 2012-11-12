; RUN: llc < %s -mcpu=generic -march=x86 | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-win32 | FileCheck %s -check-prefix=X64

; The immediate can be encoded in a smaller way if the
; instruction is a sub instead of an add.

define i32 @test1(i32 inreg %a) nounwind {
  %b = add i32 %a, 128
  ret i32 %b
; X32: subl	$-128, %eax
; X64: subl $-128, 
}
define i64 @test2(i64 inreg %a) nounwind {
  %b = add i64 %a, 2147483648
  ret i64 %b
; X32: addl	$-2147483648, %eax
; X64: subq	$-2147483648,
}
define i64 @test3(i64 inreg %a) nounwind {
  %b = add i64 %a, 128
  ret i64 %b
  
; X32: addl $128, %eax
; X64: subq	$-128,
}

define i1 @test4(i32 %v1, i32 %v2, i32* %X) nounwind {
entry:
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  store i32 0, i32* %X
  br label %overflow

overflow:
  ret i1 false
  
; X32: test4:
; X32: addl
; X32-NEXT: jo

; X64:        test4:
; X64:          addl	%e[[A1:si|dx]], %e[[A0:di|cx]]
; X64-NEXT:	jo
}

define i1 @test5(i32 %v1, i32 %v2, i32* %X) nounwind {
entry:
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %carry, label %normal

normal:
  store i32 0, i32* %X
  br label %carry

carry:
  ret i1 false

; X32: test5:
; X32: addl
; X32-NEXT: jb

; X64:        test5:
; X64:          addl	%e[[A1]], %e[[A0]]
; X64-NEXT:	jb
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32)


define i64 @test6(i64 %A, i32 %B) nounwind {
        %tmp12 = zext i32 %B to i64             ; <i64> [#uses=1]
        %tmp3 = shl i64 %tmp12, 32              ; <i64> [#uses=1]
        %tmp5 = add i64 %tmp3, %A               ; <i64> [#uses=1]
        ret i64 %tmp5

; X32: test6:
; X32:	    movl 12(%esp), %edx
; X32-NEXT: addl 8(%esp), %edx
; X32-NEXT: movl 4(%esp), %eax
; X32-NEXT: ret
        
; X64: test6:
; X64:	shlq	$32, %r[[A1]]
; X64:	leaq	(%r[[A1]],%r[[A0]]), %rax
; X64:	ret
}

define {i32, i1} @test7(i32 %v1, i32 %v2) nounwind {
   %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
   ret {i32, i1} %t
}

; X64: test7:
; X64: addl %e[[A1]], %e
; X64-NEXT: setb %dl
; X64: ret

; PR5443
define {i64, i1} @test8(i64 %left, i64 %right) nounwind {
entry:
    %extleft = zext i64 %left to i65
    %extright = zext i64 %right to i65
    %sum = add i65 %extleft, %extright
    %res.0 = trunc i65 %sum to i64
    %overflow = and i65 %sum, -18446744073709551616
    %res.1 = icmp ne i65 %overflow, 0
    %final0 = insertvalue {i64, i1} undef, i64 %res.0, 0
    %final1 = insertvalue {i64, i1} %final0, i1 %res.1, 1
    ret {i64, i1} %final1
}

; X64: test8:
; X64: addq
; X64-NEXT: sbbq
; X64-NEXT: testb

define i32 @test9(i32 %x, i32 %y) nounwind readnone {
  %cmp = icmp eq i32 %x, 10
  %sub = sext i1 %cmp to i32
  %cond = add i32 %sub, %y
  ret i32 %cond
; X64: test9:
; X64: cmpl $10
; X64: sete
; X64: subl
; X64: ret
}

define i1 @test10(i32 %x) nounwind {
entry:
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %x, i32 1)
  %obit = extractvalue {i32, i1} %t, 1
  ret i1 %obit

; X32: test10:
; X32: incl
; X32-NEXT: seto

; X64: test10:
; X64: incl
; X64-NEXT: seto
}
