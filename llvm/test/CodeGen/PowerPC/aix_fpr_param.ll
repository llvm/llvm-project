; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefix=32BIT %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefix=64BIT %s

@f1 = global float 0.000000e+00, align 4
@d1 = global double 0.000000e+00, align 8

define void @call_test_float() {
entry:
; 32BIT: renamable $r3 = LWZtoc @f1, $r2 :: (load 4 from got)
; 32BIT: renamable $f1 = LFS 0, killed renamable $r3 :: (dereferenceable load 4 from @f1)
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: BL_NOP <mcsymbol .test_float>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: renamable $x3 = LDtoc @f1, $x2 :: (load 8 from got)
; 64BIT: renamable $f1 = LFS 0, killed renamable $x3 :: (dereferenceable load 4 from @f1)
; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: BL8_NOP <mcsymbol .test_float>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  %0 = load float, float* @f1, align 4
  call void @test_float(float %0)
  ret void
}

declare void @test_float(float)

define void @call_test_floats() {
entry:
; 32BIT: renamable $r3 = LWZtoc @f1, $r2 :: (load 4 from got)
; 32BIT: renamable $f1 = LFS 0, killed renamable $r3 :: (dereferenceable load 4 from @f1)
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $f2 = COPY renamable $f1
; 32BIT: $f3 = COPY renamable $f1
; 32BIT: BL_NOP <mcsymbol .test_floats>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: renamable $x3 = LDtoc @f1, $x2 :: (load 8 from got)
; 64BIT: renamable $f1 = LFS 0, killed renamable $x3 :: (dereferenceable load 4 from @f1)
; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $f2 = COPY renamable $f1
; 64BIT: $f3 = COPY renamable $f1
; 64BIT: BL8_NOP <mcsymbol .test_floats>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  %0 = load float, float* @f1, align 4
  call void @test_floats(float %0, float %0, float %0)
  ret void
}

declare void @test_floats(float, float, float)

define void @call_test_double() {
entry:
; 32BIT: renamable $r3 = LWZtoc @d1, $r2 :: (load 4 from got)
; 32BIT: renamable $f1 = LFD 0, killed renamable $r3 :: (dereferenceable load 8 from @d1)
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: BL_NOP <mcsymbol .test_double>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: renamable $x3 = LDtoc @d1, $x2 :: (load 8 from got)
; 64BIT: renamable $f1 = LFD 0, killed renamable $x3 :: (dereferenceable load 8 from @d1)
; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: BL8_NOP <mcsymbol .test_double>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  %0 = load double, double* @d1, align 8
  call void @test_double(double %0)
  ret void
}

declare void @test_double(double)

define void @call_test_fpr_max() {
entry:
; 32BIT: renamable $r3 = LWZtoc @d1, $r2 :: (load 4 from got)
; 32BIT: renamable $f1 = LFD 0, killed renamable $r3 :: (dereferenceable load 8 from @d1)
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $f2 = COPY renamable $f1
; 32BIT: $f3 = COPY renamable $f1
; 32BIT: $f4 = COPY renamable $f1
; 32BIT: $f5 = COPY renamable $f1
; 32BIT: $f6 = COPY renamable $f1
; 32BIT: $f7 = COPY renamable $f1
; 32BIT: $f8 = COPY renamable $f1
; 32BIT: $f9 = COPY renamable $f1
; 32BIT: $f10 = COPY renamable $f1
; 32BIT: $f11 = COPY renamable $f1
; 32BIT: $f12 = COPY renamable $f1
; 32BIT: $f13 = COPY renamable $f1
; 32BIT: BL_NOP <mcsymbol .test_fpr_max>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit killed $f4, implicit killed $f5, implicit killed $f6, implicit killed $f7, implicit killed $f8, implicit killed $f9, implicit killed $f10, implicit killed $f11, implicit killed $f12, implicit killed $f13, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: renamable $x3 = LDtoc @d1, $x2 :: (load 8 from got)
; 64BIT: renamable $f1 = LFD 0, killed renamable $x3 :: (dereferenceable load 8 from @d1)
; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $f2 = COPY renamable $f1
; 64BIT: $f3 = COPY renamable $f1
; 64BIT: $f4 = COPY renamable $f1
; 64BIT: $f5 = COPY renamable $f1
; 64BIT: $f6 = COPY renamable $f1
; 64BIT: $f7 = COPY renamable $f1
; 64BIT: $f8 = COPY renamable $f1
; 64BIT: $f9 = COPY renamable $f1
; 64BIT: $f10 = COPY renamable $f1
; 64BIT: $f11 = COPY renamable $f1
; 64BIT: $f12 = COPY renamable $f1
; 64BIT: $f13 = COPY renamable $f1
; 64BIT: BL8_NOP <mcsymbol .test_fpr_max>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit killed $f4, implicit killed $f5, implicit killed $f6, implicit killed $f7, implicit killed $f8, implicit killed $f9, implicit killed $f10, implicit killed $f11, implicit killed $f12, implicit killed $f13, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  %0 = load double, double* @d1, align 8
  call void @test_fpr_max(double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0)
  ret void
}

declare void @test_fpr_max(double, double, double, double, double, double, double, double, double, double, double, double, double)

define void @call_test_mix() {
entry:
; 32BIT: renamable $r3 = LWZtoc @f1, $r2 :: (load 4 from got)
; 32BIT: renamable $r4 = LWZtoc @d1, $r2 :: (load 4 from got)
; 32BIT: renamable $f1 = LFS 0, killed renamable $r3 :: (dereferenceable load 4 from @f1)
; 32BIT: renamable $f2 = LFD 0, killed renamable $r4 :: (dereferenceable load 8 from @d1)
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r4 = LI 1
; 32BIT: $r7 = LI 97
; 32BIT: BL_NOP <mcsymbol .test_mix>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit $r4, implicit $f2, implicit killed $r7, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: renamable $x3 = LDtoc @f1, $x2 :: (load 8 from got)
; 64BIT: renamable $x4 = LDtoc @d1, $x2 :: (load 8 from got)
; 64BIT: renamable $f1 = LFS 0, killed renamable $x3 :: (dereferenceable load 4 from @f1)
; 64BIT: renamable $f2 = LFD 0, killed renamable $x4 :: (dereferenceable load 8 from @d1)
; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x4 = LI8 1
; 64BIT: $x6 = LI8 97
; 64BIT: BL8_NOP <mcsymbol .test_mix>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit $x4, implicit $f2, implicit killed $x6, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  %0 = load float, float* @f1, align 4
  %1 = load double, double* @d1, align 8
  call void @test_mix(float %0, i32 1, double %1, i8 signext 97)
  ret void
}

declare void @test_mix(float, i32, double, i8 signext)
