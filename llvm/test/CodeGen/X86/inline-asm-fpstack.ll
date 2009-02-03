; RUN: llvm-as < %s | llc -march=x86

define x86_fp80 @test1() {
        %tmp85 = call x86_fp80 asm sideeffect "fld0", "={st(0)}"()
        ret x86_fp80 %tmp85
}

define double @test2() {
        %tmp85 = call double asm sideeffect "fld0", "={st(0)}"()
        ret double %tmp85
}

define void @test3(x86_fp80 %X) {
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( x86_fp80 %X)
        ret void
}

define void @test4(double %X) {
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( double %X)
        ret void
}

define void @test5(double %X) {
        %Y = add double %X, 123.0
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( double %Y)
        ret void
}

define void @test6(double %A, double %B, double %C, 
                   double %D, double %E) nounwind  {
entry:
	; Uses the same value twice, should have one fstp after the asm.
	tail call void asm sideeffect "foo $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %A, double %A ) nounwind 
	; Uses two different values, should be in st(0)/st(1) and both be popped.
	tail call void asm sideeffect "bar $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %B, double %C ) nounwind 
	; Uses two different values, one of which isn't killed in this asm, it
	; should not be popped after the asm.
	tail call void asm sideeffect "baz $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %D, double %E ) nounwind 
	; This is the last use of %D, so it should be popped after.
	tail call void asm sideeffect "baz $0", "f,~{dirflag},~{fpsr},~{flags}"( double %D ) nounwind 
	ret void
}

