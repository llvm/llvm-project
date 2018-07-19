; RUN: llc -verify-machineinstrs < %s -mtriple=arm-eabi -mattr=v7,neon | FileCheck --check-prefix=CHECK --check-prefix=CHECK-LE %s
; RUN: llc -verify-machineinstrs < %s -mtriple=armeb-eabi -mattr=v7,neon | FileCheck --check-prefix=CHECK --check-prefix=CHECK-BE %s

@var32 = global i32 0
@vardouble = global double 0.0

define void @arg_longint( i64 %val ) {
; CHECK-LABEL: arg_longint:
; CHECK-LE: str r0, [r1]
; CHECK-BE: str r1, [r0]
   %tmp = trunc i64 %val to i32 
   store i32 %tmp, i32* @var32
   ret void
}

define void @arg_double( double %val ) {
; CHECK-LABEL: arg_double:
; CHECK: strd r0, r1, [r2]
    store double  %val, double* @vardouble
    ret void
}

define void @arg_v4i32(<4 x i32> %vec ) {
; CHECK-LABEL: arg_v4i32:
; CHECK-LE: vmov {{d[0-9]+}}, r2, r3
; CHECK-LE: vmov [[ARG_V4I32_REG:d[0-9]+]], r0, r1
; CHECK-BE: vmov {{d[0-9]+}}, r3, r2
; CHECK-BE: vmov [[ARG_V4I32_REG:d[0-9]+]], r1, r0
; CHECK: vst1.32 {[[ARG_V4I32_REG]][0]}, [r0:32]
    %tmp = extractelement <4 x i32> %vec, i32 0
    store i32 %tmp, i32* @var32
    ret void
}

define void @arg_v2f64(<2 x double> %vec ) {
; CHECK-LABEL: arg_v2f64:
; CHECK: strd r0, r1, [r2]
    %tmp = extractelement <2 x double> %vec, i32 0
    store double %tmp, double* @vardouble
    ret void
}

define i64 @return_longint() {
; CHECK-LABEL: return_longint:
; CHECK-LE: mov r0, #42
; CHECK-LE: mov r1, #0
; CHECK-BE: mov r0, #0
; CHECK-BE: mov r1, #42
    ret i64 42
}

define double @return_double() {
; CHECK-LABEL: return_double:
; CHECK-LE: vmov r0, r1, {{d[0-9]+}}
; CHECK-BE: vmov r1, r0, {{d[0-9]+}}
    ret double 1.0
}

define <4 x i32> @return_v4i32() {
; CHECK-LABEL: return_v4i32:
; CHECK-LE: vmov r0, r1, {{d[0-9]+}}
; CHECK-LE: vmov r2, r3, {{d[0-9]+}}
; CHECK-BE: vmov r1, r0, {{d[0-9]+}}
; CHECK-BE: vmov r3, r2, {{d[0-9]+}}
   ret < 4 x i32> < i32 42, i32 43, i32 44, i32 45 >
}

define <2 x double> @return_v2f64() {
; CHECK-LABEL: return_v2f64:
; CHECK-LE: vmov r0, r1, {{d[0-9]+}}
; CHECK-LE: vmov r2, r3, {{d[0-9]+}}
; CHECK-BE: vmov r1, r0, {{d[0-9]+}}
; CHECK-BE: vmov r3, r2, {{d[0-9]+}}
   ret <2 x double> < double 3.14, double 6.28 >
}

define void @caller_arg_longint() {
; CHECK-LABEL: caller_arg_longint:
; CHECK-LE: mov r0, #42
; CHECK-LE: mov r1, #0
; CHECK-BE: mov r0, #0
; CHECK-BE: mov r1, #42
   call void @arg_longint( i64 42 )
   ret void
}

define void @caller_arg_double() {
; CHECK-LABEL: caller_arg_double:
; CHECK-LE: vmov r0, r1, {{d[0-9]+}}
; CHECK-BE: vmov r1, r0, {{d[0-9]+}}
   call void @arg_double( double 1.0 )
   ret void
}

define void @caller_return_longint() {
; CHECK-LABEL: caller_return_longint:
; CHECK-LE: str r0, [r1]
; CHECK-BE: str r1, [r0]
   %val = call i64 @return_longint()
   %tmp = trunc i64 %val to i32 
   store i32 %tmp, i32* @var32
   ret void
}

define void @caller_return_double() {
; CHECK-LABEL: caller_return_double:
; CHECK-LE: vmov {{d[0-9]+}}, r0, r1
; CHECK-BE: vmov {{d[0-9]+}}, r1, r0
  %val = call double @return_double( )
  %tmp = fadd double %val, 3.14
  store double  %tmp, double* @vardouble
  ret void
}

define void @caller_return_v2f64() {
; CHECK-LABEL: caller_return_v2f64:
; CHECK: strd r0, r1, [r2]
   %val = call <2 x double> @return_v2f64( )
   %tmp = extractelement <2 x double> %val, i32 0
    store double %tmp, double* @vardouble
    ret void
}
