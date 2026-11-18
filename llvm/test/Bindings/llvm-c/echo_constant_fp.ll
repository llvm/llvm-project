; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Exercise cloning of ConstantFP initializers and instruction operands.

@fp32 = global float -2.500000e+00
@fp64 = global double 3.125000e+00
@fpnegzero = global double -0.000000e+00
@fp80_precise = global x86_fp80 0xK3FFF8000000000000001

define float @echo_const_fp32(float %x) {
entry:
  %sum = fadd float %x, 1.500000e+00
  ret float %sum
}

define double @echo_const_fp64(double %x) {
entry:
  %prod = fmul double %x, 3.141593e+00
  ret double %prod
}

define x86_fp80 @echo_const_fp80_precise() {
entry:
  ret x86_fp80 0xK3FFF8000000000000001
}
