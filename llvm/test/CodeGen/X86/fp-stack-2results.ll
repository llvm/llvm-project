; RUN: llvm-as < %s | llc -march=x86 | grep fldz
; RUN: llvm-as < %s | llc -march=x86-64 | grep fld1

; This is basically this code on x86-64:
; _Complex long double test() { return 1.0; }
define {x86_fp80, x86_fp80} @test() {
  %A = fpext double 1.0 to x86_fp80
  %B = fpext double 0.0 to x86_fp80
  ret x86_fp80 %A, x86_fp80 %B
}


;_test2:
;	fld1
;	fld	%st(0)
;	ret
define {x86_fp80, x86_fp80} @test2() {
  %A = fpext double 1.0 to x86_fp80
  ret x86_fp80 %A, x86_fp80 %A
}

; Uses both values.
define void @call1(x86_fp80 *%P1, x86_fp80 *%P2) {
  %a = call {x86_fp80,x86_fp80} @test()
  %b = getresult {x86_fp80,x86_fp80} %a, 0
  store x86_fp80 %b, x86_fp80* %P1

  %c = getresult {x86_fp80,x86_fp80} %a, 1
  store x86_fp80 %c, x86_fp80* %P2
  ret void 
}

; Uses both values, requires fxch
define void @call2(x86_fp80 *%P1, x86_fp80 *%P2) {
  %a = call {x86_fp80,x86_fp80} @test()
  %b = getresult {x86_fp80,x86_fp80} %a, 1
  store x86_fp80 %b, x86_fp80* %P1

  %c = getresult {x86_fp80,x86_fp80} %a, 0
  store x86_fp80 %c, x86_fp80* %P2
  ret void
}

; Uses ST(0), ST(1) is dead but must be popped.
define void @call3(x86_fp80 *%P1, x86_fp80 *%P2) {
  %a = call {x86_fp80,x86_fp80} @test()
  %b = getresult {x86_fp80,x86_fp80} %a, 0
  store x86_fp80 %b, x86_fp80* %P1
  ret void 
}

; Uses ST(1), ST(0) is dead and must be popped.
define void @call4(x86_fp80 *%P1, x86_fp80 *%P2) {
  %a = call {x86_fp80,x86_fp80} @test()

  %c = getresult {x86_fp80,x86_fp80} %a, 1
  store x86_fp80 %c, x86_fp80* %P2
  ret void 
}

