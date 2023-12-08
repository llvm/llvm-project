; Test the passing of scalar values in GPRs, FPRs in 64-bit calls on z/OS.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z10 | FileCheck %s

; CHECK-LABEL: call_char:
; CHECK: lghi  1, 8
define i8 @call_char(){
  %retval = call i8 (i8) @pass_char(i8 8)
  ret i8 %retval
}

; CHECK-LABEL: call_short:
; CHECK: lghi  1, 16
define i16 @call_short() {
entry:
  %retval = call i16 (i16) @pass_short(i16 16)
  ret i16 %retval
}

; CHECK-LABEL: call_int:
; CHECK: lghi  1, 32
; CHECK: lghi  2, 33
define i32 @call_int() {
entry:
  %retval = call i32 (i32, i32) @pass_int(i32 32, i32 33)
  ret i32 %retval
}

; CHECK-LABEL: call_long:
; CHECK: lghi  1, 64
; CHECK: lghi  2, 65
; CHECK: lghi  3, 66
define i64 @call_long() {
entry:
  %retval = call i64 (i64, i64, i64) @pass_long(i64 64, i64 65, i64 66)
  ret i64 %retval
}

; CHECK-LABEL: call_ptr:
; CHECK: lgr 1, 2
define i32 @call_ptr(ptr %p1, ptr %p2) {
entry:
  %retval = call i32 (ptr) @pass_ptr(ptr %p2)
  ret i32 %retval
}

; CHECK-LABEL: call_integrals:
; CHECK: lghi  1, 64
; CHECK: lghi  2, 32
; CHECK: lghi  3, 16
define i64 @call_integrals() {
entry:
  %retval = call i64 (i64, i32, i16, i64) @pass_integrals0(i64 64, i32 32, i16 16, i64 128)
  ret i64 %retval
}

; CHECK-LABEL: pass_char:
; CHECK: lgr 3, 1
define signext i8 @pass_char(i8 signext %arg) {
entry:
  ret i8 %arg
}

; CHECK-LABEL: pass_short:
; CHECK: lgr 3, 1
define signext i16 @pass_short(i16 signext %arg) {
entry:
  ret i16 %arg
}

; CHECK-LABEL: pass_int:
; CHECK: lgr 3, 2
define signext i32 @pass_int(i32 signext %arg0, i32 signext %arg1) {
entry:
  ret i32 %arg1
}

; CHECK-LABEL: pass_long:
; CHECK: agr 1, 2
; CHECK: agr 3, 1
define signext i64 @pass_long(i64 signext %arg0, i64 signext %arg1, i64 signext %arg2) {
entry:
  %N = add i64 %arg0, %arg1
  %M = add i64 %N, %arg2
  ret i64 %M
}

; CHECK-LABEL: pass_integrals0:
; CHECK: ag  2, 2200(4)
; CHECK-NEXT: lgr 3, 2
define signext i64 @pass_integrals0(i64 signext %arg0, i32 signext %arg1, i16 signext %arg2, i64 signext %arg3) {
entry:
  %N = sext i32 %arg1 to i64
  %M = add i64 %arg3, %N
  ret i64 %M
}

; CHECK-LABEL: call_float:
; CHECK: le 0, 0({{[0-9]}})
define float @call_float() {
entry:
  %ret = call float (float) @pass_float(float 0x400921FB60000000)
  ret float %ret
}

; CHECK-LABEL: call_double:
; CHECK: larl  [[GENREG:[0-9]+]], @{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT: ld  0, 0([[GENREG]])
define double @call_double() {
entry:
  %ret = call double (double) @pass_double(double 3.141000e+00)
  ret double %ret
}

; CHECK-LABEL: call_longdouble:
; CHECK: larl  [[GENREG:[0-9]+]], @{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT: ld  0, 0([[GENREG]])
; CHECK-NEXT: ld  2, 8([[GENREG]])
define fp128 @call_longdouble() {
entry:
  %ret = call fp128 (fp128) @pass_longdouble(fp128 0xLE0FC1518450562CD4000921FB5444261)
  ret fp128 %ret
}

; CHECK-LABEL: call_floats0
; CHECK: larl  [[GENREG:[0-9]+]], @{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT: ld  1, 0([[GENREG]])
; CHECK-NEXT: ld  3, 8([[GENREG]])
; CHECK: lxr 5, 0
; CHECK: lxr 0, 1
; CHECK: lxr 4, 5
define i64 @call_floats0(fp128 %arg0, double %arg1) {
entry:
  %ret = call i64 (fp128, fp128, double) @pass_floats0(fp128 0xLE0FC1518450562CD4000921FB5444261, fp128 %arg0, double %arg1)
  ret i64 %ret
}

; CHECK-LABEL: call_floats1
; CHECK: lxr 1, 0
; CHECK: ldr 0, 4
; CHECK: lxr 4, 1
define i64 @call_floats1(fp128 %arg0, double %arg1) {
entry:
  %ret = call i64 (double, fp128) @pass_floats1(double %arg1, fp128 %arg0)
  ret i64 %ret
}

; CHECK-LABEL: pass_float:
; CHECK: larl  1, @{{CPI[0-9]+_[0-9]+}}
; CHECK: aeb 0, 0(1)
define float @pass_float(float %arg) {
entry:
  %X = fadd float %arg, 0x400821FB60000000
  ret float %X
}

; CHECK-LABEL: pass_double:
; CHECK: larl  1, @{{CPI[0-9]+_[0-9]+}}
; CHECK: adb 0, 0(1)
define double @pass_double(double %arg) {
entry:
  %X = fadd double %arg, 1.414213e+00
  ret double %X
}

; CHECK-LABEL: pass_longdouble
; CHECK: larl  1, @{{CPI[0-9]+_[0-9]+}}
; CHECK: lxdb  1, 0(1)
; CHECK: axbr  0, 1
define fp128 @pass_longdouble(fp128 %arg) {
entry:
  %X = fadd fp128 %arg, 0xL10000000000000004000921FB53C8D4F
  ret fp128 %X
}

; CHECK-LABEL: pass_floats0
; CHECK: larl  1, @{{CPI[0-9]+_[0-9]+}}
; CHECK: axbr  0, 4
; CHECK: axbr  1, 0
; CHECK: cxbr  1, 5
define i64 @pass_floats0(fp128 %arg0, fp128 %arg1, double %arg2) {
  %X = fadd fp128 %arg0, %arg1
  %arg2_ext = fpext double %arg2 to fp128
  %Y = fadd fp128 %X, %arg2_ext
  %ret_bool = fcmp ueq fp128 %Y, 0xLE0FC1518450562CD4000921FB5444261
  %ret = sext i1 %ret_bool to i64
  ret i64 %ret
}

declare i64 @pass_floats1(double %arg0, fp128 %arg1)
declare i32 @pass_ptr(ptr %arg)
