; RUN: llc < %s -mattr=-avx -fast-isel -mcpu=core2 -O0 -regalloc=fast -asm-verbose=0 -fast-isel-abort=1 | FileCheck %s
; RUN: llc < %s -mattr=-avx -fast-isel -mcpu=core2 -O0 -regalloc=fast -asm-verbose=0 -fast-isel-verbose 2>&1 >/dev/null | FileCheck %s --check-prefix=STDERR --allow-empty
; RUN: llc < %s -mattr=+avx -fast-isel -mcpu=core2 -O0 -regalloc=fast -asm-verbose=0 -fast-isel-abort=1 | FileCheck %s --check-prefix=AVX

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Make sure that fast-isel folds the immediate into the binop even though it
; is non-canonical.
define i32 @test1(i32 %i) nounwind ssp {
  %and = and i32 8, %i
  ret i32 %and
}

; CHECK-LABEL: test1:
; CHECK: andl	$8, 


; rdar://9289512 - The load should fold into the compare.
define void @test2(i64 %x) nounwind ssp {
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, i64* %x.addr, align 8
  %tmp = load i64, i64* %x.addr, align 8
  %cmp = icmp sgt i64 %tmp, 42
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
; CHECK-LABEL: test2:
; CHECK: movq	%rdi, -8(%rsp)
; CHECK: cmpq	$42, -8(%rsp)
}




@G = external global i32
define i64 @test3() nounwind {
  %A = ptrtoint i32* @G to i64
  ret i64 %A
; CHECK-LABEL: test3:
; CHECK: movq _G@GOTPCREL(%rip), %rax
; CHECK-NEXT: ret
}



; rdar://9289558
@rtx_length = external global [153 x i8]

define i32 @test4(i64 %idxprom9) nounwind {
  %arrayidx10 = getelementptr inbounds [153 x i8], [153 x i8]* @rtx_length, i32 0, i64 %idxprom9
  %tmp11 = load i8, i8* %arrayidx10, align 1
  %conv = zext i8 %tmp11 to i32
  ret i32 %conv

; CHECK-LABEL: test4:
; CHECK: movq	_rtx_length@GOTPCREL(%rip), %rax
; CHECK-NEXT: movzbl	(%rax,%rdi), %eax
; CHECK-NEXT: ret
}


; PR3242 - Out of range shifts should not be folded by fastisel.
define void @test5(i32 %x, i32* %p) nounwind {
  %y = ashr i32 %x, 50000
  store i32 %y, i32* %p
  ret void

; CHECK-LABEL: test5:
; CHECK: movl	$50000, %ecx
; CHECK: sarl	%cl, %edi
; CHECK: ret
}

; rdar://9289501 - fast isel should fold trivial multiplies to shifts.
define i64 @test6(i64 %x) nounwind ssp {
entry:
  %mul = mul nsw i64 %x, 8
  ret i64 %mul

; CHECK-LABEL: test6:
; CHECK: shlq	$3, %rdi
}

define i32 @test7(i32 %x) nounwind ssp {
entry:
  %mul = mul nsw i32 %x, 8
  ret i32 %mul
; CHECK-LABEL: test7:
; CHECK: shll	$3, %edi
}


; rdar://9289507 - folding of immediates into 64-bit operations.
define i64 @test8(i64 %x) nounwind ssp {
entry:
  %add = add nsw i64 %x, 7
  ret i64 %add

; CHECK-LABEL: test8:
; CHECK: addq	$7, %rdi
}

define i64 @test9(i64 %x) nounwind ssp {
entry:
  %add = mul nsw i64 %x, 7
  ret i64 %add
; CHECK-LABEL: test9:
; CHECK: imulq	$7, %rdi, %rax
}

; rdar://9297011 - Don't reject udiv by a power of 2.
define i32 @test10(i32 %X) nounwind {
  %Y = udiv i32 %X, 8
  ret i32 %Y
; CHECK-LABEL: test10:
; CHECK: shrl	$3, 
}

define i32 @test11(i32 %X) nounwind {
  %Y = sdiv exact i32 %X, 8
  ret i32 %Y
; CHECK-LABEL: test11:
; CHECK: sarl	$3, 
}


; rdar://9297006 - Trunc to bool.
define void @test12(i8 %tmp) nounwind ssp noredzone {
entry:
  %tobool = trunc i8 %tmp to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @test12(i8 0) noredzone
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
; CHECK-LABEL: test12:
; CHECK: testb	$1,
; CHECK-NEXT: je L
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq
}

declare void @test13f(i1 %X)

define void @test13() nounwind {
  call void @test13f(i1 0)
  ret void
; CHECK-LABEL: test13:
; CHECK: xorl %edi, %edi
; CHECK-NEXT: callq
}



; rdar://9297003 - fast isel bails out on all functions taking bools
define void @test14(i8 %tmp) nounwind ssp noredzone {
entry:
  %tobool = trunc i8 %tmp to i1
  call void @test13f(i1 zeroext %tobool) noredzone
  ret void
; CHECK-LABEL: test14:
; CHECK: andb	$1, 
; CHECK: callq
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)

; rdar://9289488 - fast-isel shouldn't bail out on llvm.memcpy
define void @test15(i8* %a, i8* %b) nounwind {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 4, i32 4, i1 false)
  ret void
; CHECK-LABEL: test15:
; CHECK-NEXT: movl	(%rsi), %eax
; CHECK-NEXT: movl	%eax, (%rdi)
; CHECK-NEXT: ret
}

; Handling for varargs calls
declare void @test16callee(...) nounwind
define void @test16() nounwind {
; CHECK-LABEL: test16:
; CHECK: movl $1, %edi
; CHECK: movb $0, %al
; CHECK: callq _test16callee
  call void (...) @test16callee(i32 1)
  br label %block2

block2:
; CHECK: movsd LCP{{.*}}_{{.*}}(%rip), %xmm0
; CHECK: movb $1, %al
; CHECK: callq _test16callee

; AVX: vmovsd LCP{{.*}}_{{.*}}(%rip), %xmm0
; AVX: movb $1, %al
; AVX: callq _test16callee
  call void (...) @test16callee(double 1.000000e+00)
  ret void
}


declare void @foo() unnamed_addr ssp align 2

; Verify that we don't fold the load into the compare here.  That would move it
; w.r.t. the call.
define i32 @test17(i32 *%P) ssp nounwind {
entry:
  %tmp = load i32, i32* %P
  %cmp = icmp ne i32 %tmp, 5
  call void @foo()
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  ret i32 1

if.else:                                          ; preds = %entry
  ret i32 2
; CHECK-LABEL: test17:
; CHECK: movl	(%rdi), %eax
; CHECK: callq _foo
; CHECK: cmpl	$5, %eax
; CHECK-NEXT: je 
}

; Check that 0.0 is materialized using xorps
define void @test18(float* %p1) {
  store float 0.0, float* %p1
  ret void
; CHECK-LABEL: test18:
; CHECK: xorps
}

; Without any type hints, doubles use the smaller xorps instead of xorpd.
define void @test19(double* %p1) {
  store double 0.0, double* %p1
  ret void
; CHECK-LABEL: test19:
; CHECK: xorps
}

; Check that we fast-isel sret
%struct.a = type { i64, i64, i64 }
define void @test20() nounwind ssp {
entry:
  %tmp = alloca %struct.a, align 8
  call void @test20sret(%struct.a* sret %tmp)
  ret void
; CHECK-LABEL: test20:
; CHECK: movq %rsp, %rdi
; CHECK: callq _test20sret
}
declare void @test20sret(%struct.a* sret)

; Check that -0.0 is not materialized using xor
define void @test21(double* %p1) {
  store double -0.0, double* %p1
  ret void
; CHECK-LABEL: test21:
; CHECK-NOT: xor
; CHECK: movsd	LCPI
}

; Check that immediate arguments to a function
; do not cause massive spilling and are used
; as immediates just before the call.
define void @test22() nounwind {
entry:
  call void @foo22(i32 0)
  call void @foo22(i32 1)
  call void @foo22(i32 2)
  call void @foo22(i32 3)
  ret void
; CHECK-LABEL: test22:
; CHECK: xorl	%edi, %edi
; CHECK: callq	_foo22
; CHECK: movl	$1, %edi
; CHECK: callq	_foo22
; CHECK: movl	$2, %edi
; CHECK: callq	_foo22
; CHECK: movl	$3, %edi
; CHECK: callq	_foo22
}

declare void @foo22(i32)

; PR13563
define void @test23(i8* noalias sret %result) {
  %a = alloca i8
  %b = call i8* @foo23()
  ret void
; CHECK-LABEL: test23:
; CHECK: call
; CHECK: movq  %rdi, %rax
; CHECK: ret
}

declare i8* @foo23()

declare void @takesi32ptr(i32* %arg)

; CHECK-LABEL: allocamaterialize
define void @allocamaterialize() {
  %a = alloca i32
; CHECK: leaq {{.*}}, %rdi
  call void @takesi32ptr(i32* %a)
  ret void
}

; STDERR-NOT: FastISel missed terminator:   ret void
; CHECK-LABEL: win64ccfun
define x86_64_win64cc void @win64ccfun(i32 %i) {
; CHECK: ret
  ret void
}
