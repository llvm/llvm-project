; RUN: llc < %s -mcpu=generic -mtriple=i386-apple-darwin | FileCheck %s

; There should be no stack manipulations between the inline asm and ret.
; CHECK: test1
; CHECK: InlineAsm End
; CHECK-NEXT: ret
define x86_fp80 @test1() {
        %tmp85 = call x86_fp80 asm sideeffect "fld0", "={st(0)}"()
        ret x86_fp80 %tmp85
}

; CHECK: test2
; CHECK: InlineAsm End
; CHECK-NEXT: ret
define double @test2() {
        %tmp85 = call double asm sideeffect "fld0", "={st(0)}"()
        ret double %tmp85
}

; Setting up argument in st(0) should be a single fld.
; CHECK: test3
; CHECK: fld
; CHECK-NEXT: InlineAsm Start
; Asm consumes stack, nothing should be popped.
; CHECK: InlineAsm End
; CHECK-NOT: fstp
; CHECK: ret
define void @test3(x86_fp80 %X) {
        call void asm sideeffect "frob ", "{st(0)},~{st},~{dirflag},~{fpsr},~{flags}"( x86_fp80 %X)
        ret void
}

; CHECK: test4
; CHECK: fld
; CHECK-NEXT: InlineAsm Start
; CHECK: InlineAsm End
; CHECK-NOT: fstp
; CHECK: ret
define void @test4(double %X) {
        call void asm sideeffect "frob ", "{st(0)},~{st},~{dirflag},~{fpsr},~{flags}"( double %X)
        ret void
}

; Same as test3/4, but using value from fadd.
; The fadd can be done in xmm or x87 regs - we don't test that.
; CHECK: test5
; CHECK: InlineAsm End
; CHECK-NOT: fstp
; CHECK: ret
define void @test5(double %X) {
        %Y = fadd double %X, 123.0
        call void asm sideeffect "frob ", "{st(0)},~{st},~{dirflag},~{fpsr},~{flags}"( double %Y)
        ret void
}

; CHECK: test6
define void @test6(double %A, double %B, double %C, 
                   double %D, double %E) nounwind  {
entry:
; Uses the same value twice, should have one fstp after the asm.
; CHECK: foo
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NOT: fstp
	tail call void asm sideeffect "foo $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %A, double %A ) nounwind 
; Uses two different values, should be in st(0)/st(1) and both be popped.
; CHECK: bar
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NEXT: fstp
	tail call void asm sideeffect "bar $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %B, double %C ) nounwind 
; Uses two different values, one of which isn't killed in this asm, it
; should not be popped after the asm.
; CHECK: baz
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NOT: fstp
	tail call void asm sideeffect "baz $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %D, double %E ) nounwind 
; This is the last use of %D, so it should be popped after.
; CHECK: baz
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NOT: fstp
; CHECK: ret
	tail call void asm sideeffect "baz $0", "f,~{dirflag},~{fpsr},~{flags}"( double %D ) nounwind 
	ret void
}

; PR4185
; Passing a non-killed value to asm in {st}.
; Make sure it is duped before.
; asm kills st(0), so we shouldn't pop anything
; CHECK: testPR4185
; CHECK: fld %st(0)
; CHECK: fistpl
; CHECK-NOT: fstp
; CHECK: fistpl
; CHECK-NOT: fstp
; CHECK: ret
; A valid alternative would be to remat the constant pool load before each
; inline asm.
define void @testPR4185() {
return:
	call void asm sideeffect "fistpl $0", "{st},~{st}"(double 1.000000e+06)
	call void asm sideeffect "fistpl $0", "{st},~{st}"(double 1.000000e+06)
	ret void
}

; Passing a non-killed value through asm in {st}.
; Make sure it is not duped before.
; Second asm kills st(0), so we shouldn't pop anything
; CHECK: testPR4185b
; CHECK-NOT: fld %st(0)
; CHECK: fistl
; CHECK-NOT: fstp
; CHECK: fistpl
; CHECK-NOT: fstp
; CHECK: ret
; A valid alternative would be to remat the constant pool load before each
; inline asm.
define void @testPR4185b() {
return:
	call void asm sideeffect "fistl $0", "{st}"(double 1.000000e+06)
	call void asm sideeffect "fistpl $0", "{st},~{st}"(double 1.000000e+06)
	ret void
}

; PR4459
; The return value from ceil must be duped before being consumed by asm.
; CHECK: testPR4459
; CHECK: ceil
; CHECK: fld %st(0)
; CHECK-NOT: fxch
; CHECK: fistpl
; CHECK-NOT: fxch
; CHECK: fstpt
; CHECK: test
define void @testPR4459(x86_fp80 %a) {
entry:
	%0 = call x86_fp80 @ceil(x86_fp80 %a)
	call void asm sideeffect "fistpl $0", "{st},~{st}"( x86_fp80 %0)
	call void @test3(x86_fp80 %0 )
        ret void
}
declare x86_fp80 @ceil(x86_fp80)

; PR4484
; test1 leaves a value on the stack that is needed after the asm.
; CHECK: testPR4484
; CHECK: test1
; CHECK-NOT: fstp
; Load %a from stack after ceil
; CHECK: fldt
; CHECK-NOT: fxch
; CHECK: fistpl
; CHECK-NOT: fstp
; Set up call to test.
; CHECK: fstpt
; CHECK: test
define void @testPR4484(x86_fp80 %a) {
entry:
	%0 = call x86_fp80 @test1()
	call void asm sideeffect "fistpl $0", "{st},~{st}"(x86_fp80 %a)
	call void @test3(x86_fp80 %0)
	ret void
}

; PR4485
; CHECK: testPR4485
define void @testPR4485(x86_fp80* %a) {
entry:
	%0 = load x86_fp80* %a, align 16
	%1 = fmul x86_fp80 %0, 0xK4006B400000000000000
	%2 = fmul x86_fp80 %1, 0xK4012F424000000000000
	tail call void asm sideeffect "fistpl $0", "{st},~{st}"(x86_fp80 %2)
	%3 = load x86_fp80* %a, align 16
	%4 = fmul x86_fp80 %3, 0xK4006B400000000000000
	%5 = fmul x86_fp80 %4, 0xK4012F424000000000000
	tail call void asm sideeffect "fistpl $0", "{st},~{st}"(x86_fp80 %5)
	ret void
}

; An input argument in a fixed position is implicitly popped by the asm only if
; the input argument is tied to an output register, or it is in the clobber list.
; The clobber list case is tested above.
;
; This doesn't implicitly pop the stack:
;
;   void fist1(long double x, int *p) {
;     asm volatile ("fistl %1" : : "t"(x), "m"(*p));
;   }
;
; CHECK: fist1
; CHECK: fldt
; CHECK: fistl (%e
; CHECK: fstp
; CHECK: ret
define void @fist1(x86_fp80 %x, i32* %p) nounwind ssp {
entry:
  tail call void asm sideeffect "fistl $1", "{st},*m,~{memory},~{dirflag},~{fpsr},~{flags}"(x86_fp80 %x, i32* %p) nounwind
  ret void
}

; Here, the input operand is tied to an output which means that is is
; implicitly popped (and then the output is implicitly pushed).
;
;   long double fist2(long double x, int *p) {
;     long double y;
;     asm ("fistl %1" : "=&t"(y) : "0"(x), "m"(*p) : "memory");
;     return y;
;   }
;
; CHECK: fist2
; CHECK: fldt
; CHECK: fistl (%e
; CHECK-NOT: fstp
; CHECK: ret
define x86_fp80 @fist2(x86_fp80 %x, i32* %p) nounwind ssp {
entry:
  %0 = tail call x86_fp80 asm "fistl $2", "=&{st},0,*m,~{memory},~{dirflag},~{fpsr},~{flags}"(x86_fp80 %x, i32* %p) nounwind
  ret x86_fp80 %0
}

; An 'f' constraint is never implicitly popped:
;
;   void fucomp1(long double x, long double y) {
;     asm volatile ("fucomp %1" : : "t"(x), "f"(y) : "st");
;   }
; CHECK: fucomp1
; CHECK: fldt
; CHECK: fldt
; CHECK: fucomp %st
; CHECK: fstp
; CHECK-NOT: fstp
; CHECK: ret
define void @fucomp1(x86_fp80 %x, x86_fp80 %y) nounwind ssp {
entry:
  tail call void asm sideeffect "fucomp $1", "{st},f,~{st},~{dirflag},~{fpsr},~{flags}"(x86_fp80 %x, x86_fp80 %y) nounwind
  ret void
}

; The 'u' constraint is only popped implicitly when clobbered:
;
;   void fucomp2(long double x, long double y) {
;     asm volatile ("fucomp %1" : : "t"(x), "u"(y) : "st");
;   }
;
;   void fucomp3(long double x, long double y) {
;     asm volatile ("fucompp %1" : : "t"(x), "u"(y) : "st", "st(1)");
;   }
;
; CHECK: fucomp2
; CHECK: fldt
; CHECK: fldt
; CHECK: fucomp %st(1)
; CHECK: fstp
; CHECK-NOT: fstp
; CHECK: ret
;
; CHECK: fucomp3
; CHECK: fldt
; CHECK: fldt
; CHECK: fucompp %st(1)
; CHECK-NOT: fstp
; CHECK: ret
define void @fucomp2(x86_fp80 %x, x86_fp80 %y) nounwind ssp {
entry:
  tail call void asm sideeffect "fucomp $1", "{st},{st(1)},~{st},~{dirflag},~{fpsr},~{flags}"(x86_fp80 %x, x86_fp80 %y) nounwind
  ret void
}
define void @fucomp3(x86_fp80 %x, x86_fp80 %y) nounwind ssp {
entry:
  tail call void asm sideeffect "fucompp $1", "{st},{st(1)},~{st},~{st(1)},~{dirflag},~{fpsr},~{flags}"(x86_fp80 %x, x86_fp80 %y) nounwind
  ret void
}

; One input, two outputs, one dead output.
%complex = type { float, float }
; CHECK: sincos1
; CHECK: flds
; CHECK-NOT: fxch
; CHECK: sincos
; CHECK-NOT: fstp
; CHECK: fstp %st(1)
; CHECK-NOT: fstp
; CHECK: ret
define float @sincos1(float %x) nounwind ssp {
entry:
  %0 = tail call %complex asm "sincos", "={st},={st(1)},0,~{dirflag},~{fpsr},~{flags}"(float %x) nounwind
  %asmresult = extractvalue %complex %0, 0
  ret float %asmresult
}

; Same thing, swapped output operands.
; CHECK: sincos2
; CHECK: flds
; CHECK-NOT: fxch
; CHECK: sincos
; CHECK-NOT: fstp
; CHECK: fstp %st(1)
; CHECK-NOT: fstp
; CHECK: ret
define float @sincos2(float %x) nounwind ssp {
entry:
  %0 = tail call %complex asm "sincos", "={st(1)},={st},1,~{dirflag},~{fpsr},~{flags}"(float %x) nounwind
  %asmresult = extractvalue %complex %0, 1
  ret float %asmresult
}

; Clobber st(0) after it was live-out/dead from the previous asm.
; CHECK: sincos3
; Load x, make a copy for the second asm.
; CHECK: flds
; CHECK: fld %st(0)
; CHECK: sincos
; Discard dead result in st(0), bring x to the top.
; CHECK: fstp %st(0)
; CHECK: fxch
; x is now in st(0) for the second asm
; CHECK: sincos
; Discard both results.
; CHECK: fstp
; CHECK: fstp
; CHECK: ret
define float @sincos3(float %x) nounwind ssp {
entry:
  %0 = tail call %complex asm sideeffect "sincos", "={st(1)},={st},1,~{dirflag},~{fpsr},~{flags}"(float %x) nounwind
  %1 = tail call %complex asm sideeffect "sincos", "={st(1)},={st},1,~{dirflag},~{fpsr},~{flags}"(float %x) nounwind
  %asmresult = extractvalue %complex %0, 0
  ret float %asmresult
}

; Pass the same value in two fixed stack slots.
; CHECK: PR10602
; CHECK: flds LCPI
; CHECK: fld %st(0)
; CHECK: fcomi %st(1), %st(0)
define i32 @PR10602() nounwind ssp {
entry:
  %0 = tail call i32 asm "fcomi $2, $1; pushf; pop $0", "=r,{st},{st(1)},~{dirflag},~{fpsr},~{flags}"(double 2.000000e+00, double 2.000000e+00) nounwind
  ret i32 %0
}
