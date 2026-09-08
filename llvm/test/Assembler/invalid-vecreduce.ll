; RUN: not opt -S < %s 2>&1 | FileCheck %s

; CHECK: intrinsic return type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got float
; CHECK-NEXT: declare float @llvm.vector.reduce.fadd.f32.f64.v2f64(double, <2 x double>)
declare float @llvm.vector.reduce.fadd.f32.f64.v2f64(double %acc, <2 x double> %in)

; CHECK: intrinsic argument 0 type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got float
; CHECK-NEXT: declare double @llvm.vector.reduce.fadd.f64.f32.v2f64(float, <2 x double>)
declare double @llvm.vector.reduce.fadd.f64.f32.v2f64(float %acc, <2 x double> %in)

; CHECK: intrinsic return type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got <2 x double>
; CHECK-NEXT: declare <2 x double> @llvm.vector.reduce.fadd.v2f64.f64.v2f64(double, <2 x double>)
declare <2 x double> @llvm.vector.reduce.fadd.v2f64.f64.v2f64(double %acc, <2 x double> %in)

; CHECK: intrinsic argument 0 type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got <2 x double>
; CHECK-NEXT: declare double @llvm.vector.reduce.fadd.f64.v2f64.v2f64(<2 x double>, <2 x double>)
declare double @llvm.vector.reduce.fadd.f64.v2f64.v2f64(<2 x double> %acc, <2 x double> %in)
