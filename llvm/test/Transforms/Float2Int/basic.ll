; RUN: opt < %s -float2int -S | FileCheck %s

;
; Positive tests
;

; CHECK-LABEL: @simple1
; CHECK:  %1 = zext i8 %a to i32
; CHECK:  %2 = add i32 %1, 1
; CHECK:  %3 = trunc i32 %2 to i16
; CHECK:  ret i16 %3
define i16 @simple1(i8 %a) {
  %1 = uitofp i8 %a to float
  %2 = fadd float %1, 1.0
  %3 = fptoui float %2 to i16
  ret i16 %3
}

; CHECK-LABEL: @simple2
; CHECK:  %1 = zext i8 %a to i32
; CHECK:  %2 = sub i32 %1, 1
; CHECK:  %3 = trunc i32 %2 to i8
; CHECK:  ret i8 %3
define i8 @simple2(i8 %a) {
  %1 = uitofp i8 %a to float
  %2 = fsub float %1, 1.0
  %3 = fptoui float %2 to i8
  ret i8 %3
}

; CHECK-LABEL: @simple3
; CHECK:  %1 = zext i8 %a to i32
; CHECK:  %2 = sub i32 %1, 1
; CHECK:  ret i32 %2
define i32 @simple3(i8 %a) {
  %1 = uitofp i8 %a to float
  %2 = fsub float %1, 1.0
  %3 = fptoui float %2 to i32
  ret i32 %3
}

; CHECK-LABEL: @cmp
; CHECK:  %1 = zext i8 %a to i32
; CHECK:  %2 = zext i8 %b to i32
; CHECK:  %3 = icmp slt i32 %1, %2
; CHECK:  ret i1 %3
define i1 @cmp(i8 %a, i8 %b) {
  %1 = uitofp i8 %a to float
  %2 = uitofp i8 %b to float
  %3 = fcmp ult float %1, %2
  ret i1 %3
}

; CHECK-LABEL: @simple4
; CHECK:  %1 = zext i32 %a to i64
; CHECK:  %2 = add i64 %1, 1
; CHECK:  %3 = trunc i64 %2 to i32
; CHECK:  ret i32 %3
define i32 @simple4(i32 %a) {
  %1 = uitofp i32 %a to double
  %2 = fadd double %1, 1.0
  %3 = fptoui double %2 to i32
  ret i32 %3
}

; CHECK-LABEL: @simple5
; CHECK:  %1 = zext i8 %a to i32
; CHECK:  %2 = zext i8 %b to i32
; CHECK:  %3 = add i32 %1, 1
; CHECK:  %4 = mul i32 %3, %2
; CHECK:  ret i32 %4
define i32 @simple5(i8 %a, i8 %b) {
  %1 = uitofp i8 %a to float
  %2 = uitofp i8 %b to float
  %3 = fadd float %1, 1.0
  %4 = fmul float %3, %2
  %5 = fptoui float %4 to i32
  ret i32 %5
}

; The two chains don't interact - failure of one shouldn't
; cause failure of the other.

; CHECK-LABEL: @multi1
; CHECK:  %1 = zext i8 %a to i32
; CHECK:  %2 = zext i8 %b to i32
; CHECK:  %fc = uitofp i8 %c to float
; CHECK:  %x1 = add i32 %1, %2
; CHECK:  %z = fadd float %fc, %d
; CHECK:  %w = fptoui float %z to i32
; CHECK:  %r = add i32 %x1, %w
; CHECK:  ret i32 %r
define i32 @multi1(i8 %a, i8 %b, i8 %c, float %d) {
  %fa = uitofp i8 %a to float
  %fb = uitofp i8 %b to float
  %fc = uitofp i8 %c to float
  %x = fadd float %fa, %fb
  %y = fptoui float %x to i32
  %z = fadd float %fc, %d
  %w = fptoui float %z to i32
  %r = add i32 %y, %w
  ret i32 %r
}

; CHECK-LABEL: @simple_negzero
; CHECK:  %1 = zext i8 %a to i32
; CHECK:  %2 = add i32 %1, 0
; CHECK:  %3 = trunc i32 %2 to i16
; CHECK:  ret i16 %3
define i16 @simple_negzero(i8 %a) {
  %1 = uitofp i8 %a to float
  %2 = fadd fast float %1, -0.0
  %3 = fptoui float %2 to i16
  ret i16 %3
}

; CHECK-LABEL: @simple_negative
; CHECK: %1 = sext i8 %call to i32
; CHECK: %mul1 = mul i32 %1, -3
; CHECK: %2 = trunc i32 %mul1 to i8
; CHECK: %conv3 = sext i8 %2 to i32
; CHECK: ret i32 %conv3
define i32 @simple_negative(i8 %call) {
  %conv1 = sitofp i8 %call to float
  %mul = fmul float %conv1, -3.000000e+00
  %conv2 = fptosi float %mul to i8
  %conv3 = sext i8 %conv2 to i32
  ret i32 %conv3
}

;
; Negative tests
;

; CHECK-LABEL: @neg_multi1
; CHECK:  %fa = uitofp i8 %a to float
; CHECK:  %fc = uitofp i8 %c to float
; CHECK:  %x = fadd float %fa, %fc
; CHECK:  %y = fptoui float %x to i32
; CHECK:  %z = fadd float %fc, %d
; CHECK:  %w = fptoui float %z to i32
; CHECK:  %r = add i32 %y, %w
; CHECK:  ret i32 %r
; The two chains intersect, which means because one fails, no
; transform can occur.
define i32 @neg_multi1(i8 %a, i8 %b, i8 %c, float %d) {
  %fa = uitofp i8 %a to float
  %fc = uitofp i8 %c to float
  %x = fadd float %fa, %fc
  %y = fptoui float %x to i32
  %z = fadd float %fc, %d
  %w = fptoui float %z to i32
  %r = add i32 %y, %w
  ret i32 %r
}

; CHECK-LABEL: @neg_muld
; CHECK:  %fa = uitofp i32 %a to double
; CHECK:  %fb = uitofp i32 %b to double
; CHECK:  %mul = fmul double %fa, %fb
; CHECK:  %r = fptoui double %mul to i64
; CHECK:  ret i64 %r
; The i32 * i32 = i64, which has 64 bits, which is greater than the 52 bits
; that can be exactly represented in a double.
define i64 @neg_muld(i32 %a, i32 %b) {
  %fa = uitofp i32 %a to double
  %fb = uitofp i32 %b to double
  %mul = fmul double %fa, %fb
  %r = fptoui double %mul to i64
  ret i64 %r
}

; CHECK-LABEL: @neg_mulf
; CHECK:  %fa = uitofp i16 %a to float
; CHECK:  %fb = uitofp i16 %b to float
; CHECK:  %mul = fmul float %fa, %fb
; CHECK:  %r = fptoui float %mul to i32
; CHECK:  ret i32 %r
; The i16 * i16 = i32, which can't be represented in a float, but can in a
; double. This should fail, as the written code uses floats, not doubles so
; the original result may be inaccurate.
define i32 @neg_mulf(i16 %a, i16 %b) {
  %fa = uitofp i16 %a to float
  %fb = uitofp i16 %b to float
  %mul = fmul float %fa, %fb
  %r = fptoui float %mul to i32
  ret i32 %r
}

; CHECK-LABEL: @neg_cmp
; CHECK:  %1 = uitofp i8 %a to float
; CHECK:  %2 = uitofp i8 %b to float
; CHECK:  %3 = fcmp false float %1, %2
; CHECK:  ret i1 %3
; "false" doesn't have an icmp equivalent.
define i1 @neg_cmp(i8 %a, i8 %b) {
  %1 = uitofp i8 %a to float
  %2 = uitofp i8 %b to float
  %3 = fcmp false float %1, %2
  ret i1 %3
}

; CHECK-LABEL: @neg_div
; CHECK:  %1 = uitofp i8 %a to float
; CHECK:  %2 = fdiv float %1, 1.0
; CHECK:  %3 = fptoui float %2 to i16
; CHECK:  ret i16 %3
; Division isn't a supported operator.
define i16 @neg_div(i8 %a) {
  %1 = uitofp i8 %a to float
  %2 = fdiv float %1, 1.0
  %3 = fptoui float %2 to i16
  ret i16 %3
}

; CHECK-LABEL: @neg_remainder
; CHECK:  %1 = uitofp i8 %a to float
; CHECK:  %2 = fadd float %1, 1.2
; CHECK:  %3 = fptoui float %2 to i16
; CHECK:  ret i16 %3
; 1.2 is not an integer.
define i16 @neg_remainder(i8 %a) {
  %1 = uitofp i8 %a to float
  %2 = fadd float %1, 1.25
  %3 = fptoui float %2 to i16
  ret i16 %3
}

; CHECK-LABEL: @neg_toolarge
; CHECK:  %1 = uitofp i80 %a to fp128
; CHECK:  %2 = fadd fp128 %1, %1
; CHECK:  %3 = fptoui fp128 %2 to i80
; CHECK:  ret i80 %3
; i80 > i64, which is the largest bitwidth handleable by default.
define i80 @neg_toolarge(i80 %a) {
  %1 = uitofp i80 %a to fp128
  %2 = fadd fp128 %1, %1
  %3 = fptoui fp128 %2 to i80
  ret i80 %3
}

; CHECK-LABEL: @neg_calluser
; CHECK: sitofp
; CHECK: fcmp
; The sequence %1..%3 cannot be converted because %4 uses %2.
define i32 @neg_calluser(i32 %value) {
  %1 = sitofp i32 %value to double
  %2 = fadd double %1, 1.0
  %3 = fcmp olt double %2, 0.000000e+00
  %4 = tail call double @g(double %2)
  %5 = fptosi double %4 to i32
  %6 = zext i1 %3 to i32
  %7 = add i32 %6, %5
  ret i32 %7
}
declare double @g(double)

; CHECK-LABEL: @neg_vector
; CHECK:  %1 = uitofp <4 x i8> %a to <4 x float>
; CHECK:  %2 = fptoui <4 x float> %1 to <4 x i16>
; CHECK:  ret <4 x i16> %2
define <4 x i16> @neg_vector(<4 x i8> %a) {
  %1 = uitofp <4 x i8> %a to <4 x float>
  %2 = fptoui <4 x float> %1 to <4 x i16>
  ret <4 x i16> %2
}
