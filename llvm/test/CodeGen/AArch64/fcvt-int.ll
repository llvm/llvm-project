; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define i32 @test_floattoi32(float %in) {
; CHECK-LABEL: test_floattoi32:

  %signed = fptosi float %in to i32
  %unsigned = fptoui float %in to i32
; CHECK-DAG: fcvtzu [[UNSIG:w[0-9]+]], {{s[0-9]+}}
; CHECK-DAG: fcvtzs [[SIG:w[0-9]+]], {{s[0-9]+}}

  %res = sub i32 %signed, %unsigned
; CHECK: sub {{w[0-9]+}}, [[SIG]], [[UNSIG]]

  ret i32 %res
; CHECK: ret
}

define i32 @test_doubletoi32(double %in) {
; CHECK-LABEL: test_doubletoi32:

  %signed = fptosi double %in to i32
  %unsigned = fptoui double %in to i32
; CHECK-DAG: fcvtzu [[UNSIG:w[0-9]+]], {{d[0-9]+}}
; CHECK-DAG: fcvtzs [[SIG:w[0-9]+]], {{d[0-9]+}}

  %res = sub i32 %signed, %unsigned
; CHECK: sub {{w[0-9]+}}, [[SIG]], [[UNSIG]]

  ret i32 %res
; CHECK: ret
}

define i64 @test_floattoi64(float %in) {
; CHECK-LABEL: test_floattoi64:

  %signed = fptosi float %in to i64
  %unsigned = fptoui float %in to i64
; CHECK-DAG: fcvtzu [[UNSIG:x[0-9]+]], {{s[0-9]+}}
; CHECK-DAG: fcvtzs [[SIG:x[0-9]+]], {{s[0-9]+}}

  %res = sub i64 %signed, %unsigned
; CHECK: sub {{x[0-9]+}}, [[SIG]], [[UNSIG]]

  ret i64 %res
; CHECK: ret
}

define i64 @test_doubletoi64(double %in) {
; CHECK-LABEL: test_doubletoi64:

  %signed = fptosi double %in to i64
  %unsigned = fptoui double %in to i64
; CHECK-DAG: fcvtzu [[UNSIG:x[0-9]+]], {{d[0-9]+}}
; CHECK-DAG: fcvtzs [[SIG:x[0-9]+]], {{d[0-9]+}}

  %res = sub i64 %signed, %unsigned
; CHECK: sub {{x[0-9]+}}, [[SIG]], [[UNSIG]]

  ret i64 %res
; CHECK: ret
}

define float @test_i32tofloat(i32 %in) {
; CHECK-LABEL: test_i32tofloat:

  %signed = sitofp i32 %in to float
  %unsigned = uitofp i32 %in to float
; CHECK-DAG: ucvtf [[UNSIG:s[0-9]+]], {{w[0-9]+}}
; CHECK-DAG: scvtf [[SIG:s[0-9]+]], {{w[0-9]+}}

  %res = fsub float %signed, %unsigned
; CHECK: fsub {{s[0-9]+}}, [[SIG]], [[UNSIG]]
  ret float %res
; CHECK: ret
}

define double @test_i32todouble(i32 %in) {
; CHECK-LABEL: test_i32todouble:

  %signed = sitofp i32 %in to double
  %unsigned = uitofp i32 %in to double
; CHECK-DAG: ucvtf [[UNSIG:d[0-9]+]], {{w[0-9]+}}
; CHECK-DAG: scvtf [[SIG:d[0-9]+]], {{w[0-9]+}}

  %res = fsub double %signed, %unsigned
; CHECK: fsub {{d[0-9]+}}, [[SIG]], [[UNSIG]]
  ret double %res
; CHECK: ret
}

define float @test_i64tofloat(i64 %in) {
; CHECK-LABEL: test_i64tofloat:

  %signed = sitofp i64 %in to float
  %unsigned = uitofp i64 %in to float
; CHECK-DAG: ucvtf [[UNSIG:s[0-9]+]], {{x[0-9]+}}
; CHECK-DAG: scvtf [[SIG:s[0-9]+]], {{x[0-9]+}}

  %res = fsub float %signed, %unsigned
; CHECK: fsub {{s[0-9]+}}, [[SIG]], [[UNSIG]]
  ret float %res
; CHECK: ret
}

define double @test_i64todouble(i64 %in) {
; CHECK-LABEL: test_i64todouble:

  %signed = sitofp i64 %in to double
  %unsigned = uitofp i64 %in to double
; CHECK-DAG: ucvtf [[UNSIG:d[0-9]+]], {{x[0-9]+}}
; CHECK-DAG: scvtf [[SIG:d[0-9]+]], {{x[0-9]+}}

  %res = fsub double %signed, %unsigned
; CHECK: sub {{d[0-9]+}}, [[SIG]], [[UNSIG]]
  ret double %res
; CHECK: ret
}

define i32 @test_bitcastfloattoi32(float %in) {
; CHECK-LABEL: test_bitcastfloattoi32:

   %res = bitcast float %in to i32
; CHECK: fmov {{w[0-9]+}}, {{s[0-9]+}}
   ret i32 %res
}

define i64 @test_bitcastdoubletoi64(double %in) {
; CHECK-LABEL: test_bitcastdoubletoi64:

   %res = bitcast double %in to i64
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
   ret i64 %res
}

define float @test_bitcasti32tofloat(i32 %in) {
; CHECK-LABEL: test_bitcasti32tofloat:

   %res = bitcast i32 %in to float
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
   ret float %res

}

define double @test_bitcasti64todouble(i64 %in) {
; CHECK-LABEL: test_bitcasti64todouble:

   %res = bitcast i64 %in to double
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
   ret double %res

}

define double @bitcast_fabs(double %x) {
; CHECK-LABEL: bitcast_fabs:
; CHECK:       ; BB#0:
; CHECK-NEXT:    fabs d0, d0
; CHECK-NEXT:    ret
;
  %bc1 = bitcast double %x to i64
  %and = and i64 %bc1, 9223372036854775807
  %bc2 = bitcast i64 %and to double
  ret double %bc2
}

define float @bitcast_fneg(float %x) {
; CHECK-LABEL: bitcast_fneg:
; CHECK:       ; BB#0:
; CHECK-NEXT:    fneg s0, s0
; CHECK-NEXT:    ret
;
  %bc1 = bitcast float %x to i32
  %xor = xor i32 %bc1, 2147483648
  %bc2 = bitcast i32 %xor to float
  ret float %bc2
}

