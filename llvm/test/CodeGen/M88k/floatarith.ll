; Test floating point arithmetic.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -O0 | FileCheck --check-prefixes=CHECK,MC88100 %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -O0 | FileCheck --check-prefixes=CHECK,MC88110 %s

define float @trunc(double %a) {
; CHECK-LABEL: trunc:
; MC88100: fsub.sds %r2, %r2, %r0
; MC88110: fcvt.sd %r2, %r2
; CHECK: jmp %r1
  %trnc = fptrunc double %a to float
  ret float %trnc
}

define double @extend(float %a) {
; CHECK-LABEL: extend:
; MC88100: fsub.dss %r4, %r2, %r0
; MC88110: fcvt.ds %r4, %r2
; CHECK: jmp %r1
  %ext = fpext float %a to double
  ret double %ext
}

define float @negate(float %a) {
; CHECK-LABEL: negate:
; CHECK: xor.u %r2, %r2, 32768
; CHECK: jmp %r1
  %neg = fneg float %a
  ret float %neg
}

define i32 @f32tosi32(float %a) {
; CHECK-LABEL: f32tosi32:
; CHECK: trnc.ss %r2, %r2
; CHECK: jmp %r1
  %trnc = fptosi float %a to i32
  ret i32 %trnc
}

define i32 @f64tosi32(double %a) {
; CHECK-LABEL: f64tosi32:
; CHECK: trnc.sd %r2, %r2
; CHECK: jmp %r1
  %trnc = fptosi double %a to i32
  ret i32 %trnc
}

define i64 @f32tosi64(float %a) {
; CHECK-LABEL: f32tosi64:
; CHECK: bsr __fixsfdi
; CHECK: jmp %r1
  %trnc = fptosi float %a to i64
  ret i64 %trnc
}

define i64 @f64tosi64(double %a) {
; CHECK-LABEL: f64tosi64:
; CHECK: bsr __fixdfdi
; CHECK: jmp %r1
  %trnc = fptosi double %a to i64
  ret i64 %trnc
}

define float @fadd1(float %a, float %b) {
; CHECK-LABEL: fadd1:
; CHECK: fadd.sss %r2, %r2, %r3
; CHECK: jmp %r1
  %sum = fadd float %a, %b
  ret float %sum
}

define float @fadd2(float %a, double %b) {
; CHECK-LABEL: fadd2:
; CHECK: fadd.ssd %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fadd double %ext, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fadd3(double %a, float %b) {
; CHECK-LABEL: fadd3:
; CHECK: fadd.sds %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fadd double %a, %ext
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fadd4(double %a, double %b) {
; CHECK-LABEL: fadd4:
; CHECK: fadd.sdd %r2, %r2, %r4
; CHECK: jmp %r1
  %sum = fadd double %a, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define double @fadd5(double %a, double %b) {
; CHECK-LABEL: fadd5:
; CHECK: fadd.ddd %r4, %r2, %r4
; CHECK: jmp %r1
  %sum = fadd double %a, %b
  ret double %sum
}

define double @fadd6(double %a, float %b) {
; CHECK-LABEL: fadd6:
; CHECK: fadd.dds %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fadd double %a, %ext
  ret double %sum
}

define double @fadd7(float %a, double %b) {
; CHECK-LABEL: fadd7:
; CHECK: fadd.dsd %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fadd double %ext, %b
  ret double %sum
}

define double @fadd8(float %a, float %b) {
; CHECK-LABEL: fadd8:
; CHECK: fadd.dss %r4, %r2, %r3
; CHECK: jmp %r1
  %sum = fadd float %a, %b
  %ext = fpext float %sum to double
  ret double %ext
}

define float @fsub1(float %a, float %b) {
; CHECK-LABEL: fsub1:
; CHECK: fsub.sss %r2, %r2, %r3
; CHECK: jmp %r1
  %sum = fsub float %a, %b
  ret float %sum
}

define float @fsub2(float %a, double %b) {
; CHECK-LABEL: fsub2:
; CHECK: fsub.ssd %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fsub double %ext, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fsub3(double %a, float %b) {
; CHECK-LABEL: fsub3:
; CHECK: fsub.sds %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fsub double %a, %ext
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fsub4(double %a, double %b) {
; CHECK-LABEL: fsub4:
; CHECK: fsub.sdd %r2, %r2, %r4
; CHECK: jmp %r1
  %sum = fsub double %a, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define double @fsub5(double %a, double %b) {
; CHECK-LABEL: fsub5:
; CHECK: fsub.ddd %r4, %r2, %r4
; CHECK: jmp %r1
  %sum = fsub double %a, %b
  ret double %sum
}

define double @fsub6(double %a, float %b) {
; CHECK-LABEL: fsub6:
; CHECK: fsub.dds %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fsub double %a, %ext
  ret double %sum
}

define double @fsub7(float %a, double %b) {
; CHECK-LABEL: fsub7:
; CHECK: fsub.dsd %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fsub double %ext, %b
  ret double %sum
}

define double @fsub8(float %a, float %b) {
; CHECK-LABEL: fsub8:
; CHECK: fsub.dss %r4, %r2, %r3
; CHECK: jmp %r1
  %sum = fsub float %a, %b
  %ext = fpext float %sum to double
  ret double %ext
}

define float @fmul1(float %a, float %b) {
; CHECK-LABEL: fmul1:
; CHECK: fmul.sss %r2, %r2, %r3
; CHECK: jmp %r1
  %sum = fmul float %a, %b
  ret float %sum
}

define float @fmul2(float %a, double %b) {
; CHECK-LABEL: fmul2:
; CHECK: fmul.ssd %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fmul double %ext, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fmul3(double %a, float %b) {
; CHECK-LABEL: fmul3:
; CHECK: fmul.sds %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fmul double %a, %ext
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fmul4(double %a, double %b) {
; CHECK-LABEL: fmul4:
; CHECK: fmul.sdd %r2, %r2, %r4
; CHECK: jmp %r1
  %sum = fmul double %a, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define double @fmul5(double %a, double %b) {
; CHECK-LABEL: fmul5:
; CHECK: fmul.ddd %r4, %r2, %r4
; CHECK: jmp %r1
  %sum = fmul double %a, %b
  ret double %sum
}

define double @fmul6(double %a, float %b) {
; CHECK-LABEL: fmul6:
; CHECK: fmul.dds %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fmul double %a, %ext
  ret double %sum
}

define double @fmul7(float %a, double %b) {
; CHECK-LABEL: fmul7:
; CHECK: fmul.dsd %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fmul double %ext, %b
  ret double %sum
}

define double @fmul8(float %a, float %b) {
; CHECK-LABEL: fmul8:
; CHECK: fmul.dss %r4, %r2, %r3
; CHECK: jmp %r1
  %sum = fmul float %a, %b
  %ext = fpext float %sum to double
  ret double %ext
}

define float @fdiv1(float %a, float %b) {
; CHECK-LABEL: fdiv1:
; CHECK: fdiv.sss %r2, %r2, %r3
; CHECK: jmp %r1
  %sum = fdiv float %a, %b
  ret float %sum
}

define float @fdiv2(float %a, double %b) {
; CHECK-LABEL: fdiv2:
; CHECK: fdiv.ssd %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fdiv double %ext, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fdiv3(double %a, float %b) {
; CHECK-LABEL: fdiv3:
; CHECK: fdiv.sds %r2, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fdiv double %a, %ext
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define float @fdiv4(double %a, double %b) {
; CHECK-LABEL: fdiv4:
; CHECK: fdiv.sdd %r2, %r2, %r4
; CHECK: jmp %r1
  %sum = fdiv double %a, %b
  %trnc = fptrunc double %sum to float
  ret float %trnc
}

define double @fdiv5(double %a, double %b) {
; CHECK-LABEL: fdiv5:
; CHECK: fdiv.ddd %r4, %r2, %r4
; CHECK: jmp %r1
  %sum = fdiv double %a, %b
  ret double %sum
}

define double @fdiv6(double %a, float %b) {
; CHECK-LABEL: fdiv6:
; CHECK: fdiv.dds %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %b to double
  %sum = fdiv double %a, %ext
  ret double %sum
}

define double @fdiv7(float %a, double %b) {
; CHECK-LABEL: fdiv7:
; CHECK: fdiv.dsd %r4, %r2, %r4
; CHECK: jmp %r1
  %ext = fpext float %a to double
  %sum = fdiv double %ext, %b
  ret double %sum
}

define double @fdiv8(float %a, float %b) {
; CHECK-LABEL: fdiv8:
; CHECK: fdiv.dss %r4, %r2, %r3
; CHECK: jmp %r1
  %sum = fdiv float %a, %b
  %ext = fpext float %sum to double
  ret double %ext
}
