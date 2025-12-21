; Test passing variable argument lists in 64-bit calls on z/OS.
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z14 | FileCheck %s -check-prefix=ARCH12
; CHECK-LABEL: call_vararg_double0:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 6,8(5)
; CHECK-NEXT:    lg 5,0(5)
; CHECK-NEXT:    llihf 3,1074118262
; CHECK-NEXT:    oilf 3,3367254360
; CHECK-NEXT:    lghi 1,1
; CHECK-NEXT:    lghi 2,2
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_double0() {
entry:
  %retval = call i64 (i64, i64, ...) @pass_vararg0(i64 1, i64 2, double 2.718000e+00)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_double1:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    llihf 0,1074118262
; CHECK-NEXT:    oilf 0,3367254360
; CHECK-NEXT:    lg 6,8(5)
; CHECK-NEXT:    lg 5,0(5)
; CHECK-NEXT:    llihf 3,1074340036
; CHECK-NEXT:    oilf 3,2611340116
; CHECK-NEXT:    lghi 1,1
; CHECK-NEXT:    lghi 2,2
; CHECK-NEXT:    stg 0,2200(4)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_double1() {
entry:
  %retval = call i64 (i64, i64, ...) @pass_vararg0(i64 1, i64 2, double 3.141000e+00, double 2.718000e+00)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_double2:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 6,24(5)
; CHECK-NEXT:    lg 5,16(5)
; CHECK-NEXT:    llihf 2,1074118262
; CHECK-NEXT:    oilf 2,3367254360
; CHECK-NEXT:    lghi 1,8200
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_double2() {
entry:
  %retval = call i64 (i64, ...) @pass_vararg2(i64 8200, double 2.718000e+00)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_double3:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    llihf 0,1072703839
; CHECK-NEXT:    oilf 0,2861204133
; CHECK-NEXT:    lg 6,40(5)
; CHECK-NEXT:    lg 5,32(5)
; CHECK-NEXT:    llihf 1,1074118262
; CHECK-NEXT:    oilf 1,3367254360
; CHECK-NEXT:    llihf 2,1074340036
; CHECK-NEXT:    oilf 2,2611340116
; CHECK-NEXT:    llihf 3,1073127358
; CHECK-NEXT:    oilf 3,1992864825
; CHECK-NEXT:    stg 0,2200(4)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_double3() {
entry:
  %retval = call i64 (...) @pass_vararg3(double 2.718000e+00, double 3.141000e+00, double 1.414000e+00, double 1.010101e+00)
  ret i64 %retval
}

;; TODO: The extra COPY after LGDR is unnecessary (machine-scheduler introduces the overlap).
; CHECK-LABEL: call_vararg_both0:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 6,40(5)
; CHECK-NEXT:    lg 5,32(5)
; CHECK-NEXT:    lgdr 0,0
; CHECK-NEXT:    lgr 2,1
; CHECK-NEXT:    lgr 1,0
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_both0(i64 %arg0, double %arg1) {
  %retval  = call i64(...) @pass_vararg3(double %arg1, i64 %arg0)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_long_double0:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    larl 1,L#CPI5_0
; CHECK-NEXT:    ld 0,0(1)
; CHECK-NEXT:    ld 2,8(1)
; CHECK-NEXT:    lg 6,8(5)
; CHECK-NEXT:    lg 5,0(5)
; CHECK-NEXT:    lgdr 3,0
; CHECK-NEXT:    lghi 1,1
; CHECK-NEXT:    lghi 2,2
; CHECK-NEXT:    std 0,2192(4)
; CHECK-NEXT:    std 2,2200(4)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_long_double0() {
entry:
  %retval = call i64 (i64, i64, ...) @pass_vararg0(i64 1, i64 2, fp128 0xLE0FC1518450562CD4000921FB5444261)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_long_double1:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 6,8(5)
; CHECK-NEXT:    lg 5,0(5)
; CHECK-NEXT:    lgdr 3,0
; CHECK-NEXT:    lghi 1,1
; CHECK-NEXT:    lghi 2,2
; CHECK-NEXT:    std 0,2192(4)
; CHECK-NEXT:    std 2,2200(4)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_long_double1(fp128 %arg0) {
entry:
  %retval = call i64 (i64, i64, ...) @pass_vararg0(i64 1, i64 2, fp128 %arg0)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_long_double2
; CHECK-LABEL: call_vararg_long_double2:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    std 4,2208(4)
; CHECK-NEXT:    std 6,2216(4)
; CHECK-NEXT:    lg 6,8(5)
; CHECK-NEXT:    lg 5,0(5)
; CHECK-NEXT:    lgdr 3,0
; CHECK-NEXT:    lghi 1,1
; CHECK-NEXT:    lghi 2,2
; CHECK-NEXT:    std 0,2192(4)
; CHECK-NEXT:    std 2,2200(4)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_long_double2(fp128 %arg0, fp128 %arg1) {
entry:
  %retval = call i64 (i64, i64, ...) @pass_vararg0(i64 1, i64 2, fp128 %arg0, fp128 %arg1)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_long_double3:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 6,40(5)
; CHECK-NEXT:    lg 5,32(5)
; CHECK-NEXT:    lgdr 3,2
; CHECK-NEXT:    lgdr 2,0
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_long_double3(fp128 %arg0) {
entry:
  %retval = call i64 (...) @pass_vararg3(fp128 %arg0)
  ret i64 %retval
}

; ARCH12-LABEL: call_vec_vararg_test0
; ARCH12: vlgvg 3,24,1
; ARCH12: vlgvg 2,24,0
; ARCH12: lghi  1,1
define void @call_vec_vararg_test0(<2 x double> %v) {
  %retval = call i64(i64, ...) @pass_vararg2(i64 1, <2 x double> %v)
  ret void
}

; ARCH12-LABEL: call_vec_vararg_test1
; ARCH12: larl  1,L#CPI10_0
; ARCH12: vl    0,0(1),3
; ARCH12: vlgvg 3,24,0
; ARCH12: vrepg 2,0,1
; ARCH12: vst   25,2208(4),3
; ARCH12: vst   24,2192(4),3
define void @call_vec_vararg_test1(<4 x i32> %v, <2 x i64> %w) {
  %retval = call i64(fp128, ...) @pass_vararg1(fp128 0xLE0FC1518450562CD4000921FB5444261, <4 x i32> %v, <2 x i64> %w)
  ret void
}

; ARCH12-LABEL: call_vec_char_vararg_straddle
; ARCH12: vlgvg 3,24,0
; ARCH12: lghi  1,1
; ARCH12: lghi  2,2
; ARCH12: vst   24,2192(4),3
define void @call_vec_char_vararg_straddle(<16 x i8> %v) {
  %retval = call i64(i64, i64, ...) @pass_vararg0(i64 1, i64 2, <16 x i8> %v)
  ret void
}

; ARCH12-LABEL: call_vec_short_vararg_straddle
; ARCH12: vlgvg 3,24,0
; ARCH12: lghi  1,1
; ARCH12: lghi  2,2
; ARCH12: vst   24,2192(4),3
define void @call_vec_short_vararg_straddle(<8 x i16> %v) {
  %retval = call i64(i64, i64, ...) @pass_vararg0(i64 1, i64 2, <8 x i16> %v)
  ret void
}

; ARCH12-LABEL: call_vec_int_vararg_straddle
; ARCH12: vlgvg 3,24,0
; ARCH12: lghi  1,1
; ARCH12: lghi  2,2
; ARCH12: vst 24,2192(4),3
define void @call_vec_int_vararg_straddle(<4 x i32> %v) {
  %retval = call i64(i64, i64, ...) @pass_vararg0(i64 1, i64 2, <4 x i32> %v)
  ret void
}

; ARCH12-LABEL: call_vec_double_vararg_straddle
; ARCH12: vlgvg 3,24,0
; ARCH12: lghi  1,1
; ARCH12: lghi  2,2
; ARCH12: vst 24,2192(4),3
define void @call_vec_double_vararg_straddle(<2 x double> %v) {
  %retval = call i64(i64, i64, ...) @pass_vararg0(i64 1, i64 2, <2 x double> %v)
  ret void
}

; CHECK-LABEL: call_vararg_integral0:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 0,2392(4)
; CHECK-NEXT:    lg 6,40(5)
; CHECK-NEXT:    lg 5,32(5)
; CHECK-NEXT:    stg 0,2200(4)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_integral0(i32 signext %arg0, i16 signext %arg1, i64 signext %arg2, i8 signext %arg3) {
entry:
  %retval = call i64(...) @pass_vararg3(i32 signext %arg0, i16 signext %arg1, i64 signext %arg2, i8 signext %arg3)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_float0:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 6,24(5)
; CHECK-NEXT:    lg 5,16(5)
; CHECK-NEXT:    lghi 1,1
; CHECK-NEXT:    llihf 2,1073692672
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_float0() {
entry:
  %retval = call i64 (i64, ...) @pass_vararg2(i64 1, float 1.953125)
  ret i64 %retval
}

; CHECK-LABEL: call_vararg_float1:
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lg 6,72(5)
; CHECK-NEXT:    lg 5,64(5)
; CHECK-NEXT:    larl 1,L#CPI17_0
; CHECK-NEXT:    le 0,0(1)
; CHECK-NEXT:    llihf 0,1073692672
; CHECK-NEXT:    llihh 2,16384
; CHECK-NEXT:    llihh 3,16392
; CHECK-NEXT:    stg 0,2200(4)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @call_vararg_float1() {
entry:
  %retval = call i64 (float, ...) @pass_vararg4(float 1.0, float 2.0, float 3.0, float 1.953125)
  ret i64 %retval
}

; Derived from C source:
; #define _VARARG_EXT_
; #include <stdarg.h>
;
; long pass(long x, ...) {
;   va_list va;
;   va_start(va, x);
;   long ret = va_arg(va, long);
;   va_end(va);
;   return ret;
; }
;
; CHECK-LABEL: pass_vararg:
; CHECK:         stmg 6,7,1904(4)
; CHECK-NEXT:    aghi 4,-160
; CHECK-NEXT:    stg 2,2344(4)
; CHECK-NEXT:    stg 3,2352(4)
; CHECK-NEXT:    la 0,2352(4)
; CHECK-NEXT:    stg 0,2200(4)
; CHECK-NEXT:    lg 3,2344(4)
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,160
; CHECK-NEXT:    b 2(7)
define hidden i64 @pass_vararg(i64 %x, ...) {
entry:
  %va = alloca ptr, align 8
  call void @llvm.va_start(ptr %va)
  %argp.cur = load ptr, ptr %va, align 8
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i64 8
  store ptr %argp.next, ptr %va, align 8
  %ret = load i64, ptr %argp.cur, align 8
  call void @llvm.va_end(ptr %va)
  ret i64 %ret
}

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

declare i64 @pass_vararg0(i64 %arg0, i64 %arg1, ...)
declare i64 @pass_vararg1(fp128 %arg0, ...)
declare i64 @pass_vararg2(i64 %arg0, ...)
declare i64 @pass_vararg3(...)
declare i64 @pass_vararg4(float, ...)
