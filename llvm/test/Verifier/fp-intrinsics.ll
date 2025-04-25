; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)

; Test an illegal value for the rounding mode argument.
; CHECK: invalid rounding mode argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynomic", metadata !"fpexcept.strict") #1
define double @f2(double %a, double %b) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                          double %a, double %b,
                                          metadata !"round.dynomic",
                                          metadata !"fpexcept.strict") #0
  ret double %fadd
}

; Test an illegal value for the exception behavior argument.
; CHECK-NEXT: invalid exception behavior argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.restrict") #1
define double @f3(double %a, double %b) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                        double %a, double %b,
                                        metadata !"round.dynamic",
                                        metadata !"fpexcept.restrict") #0
  ret double %fadd
}

; Test an illegal value for the rounding mode argument.
; CHECK-NEXT: invalid rounding mode argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynomic", metadata !"fpexcept.strict") #1
define double @f4(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                          double %a,
                                          metadata !"round.dynomic",
                                          metadata !"fpexcept.strict") #0
  ret double %fadd
}

; Test an illegal value for the exception behavior argument.
; CHECK-NEXT: invalid exception behavior argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynamic", metadata !"fpexcept.restrict") #1
define double @f5(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                        double %a,
                                        metadata !"round.dynamic",
                                        metadata !"fpexcept.restrict") #0
  ret double %fadd
}

; Test multiple fp.control bundles.
; CHECK-NEXT: Multiple "fp.control" operand bundles
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata !"rtz"), "fp.control"(metadata !"rtz") ]
define double @f6(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !"rtz"), "fp.control"(metadata !"rtz") ]
  ret double %ftrunc
}

; Test fp.control bundle that has more than one rounding mode specification.
; CHECK-NEXT: Rounding mode is specified more that once
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata !"rtz", metadata !"rte") ]
define double @f7(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !"rtz", metadata !"rte") ]
  ret double %ftrunc
}

; Test fp.control bundle that has non-metadata operand.
; CHECK-NEXT: Value of a "fp.control" bundle operand must be a metadata
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(i32 0) ]
define double @f8(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(i32 0) ]
  ret double %ftrunc
}

; Test fp.control bundle that has non-string operand.
; CHECK-NEXT: Value of a "fp.control" bundle operand must be a string
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata i64 3) ]
define double @f9(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !{i64 3}) ]
  ret double %ftrunc
}

; Test fp.control bundle that specifies incorrect value.
; CHECK-NEXT: Unrecognized value in "fp.control" bundle operand
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata !"qqq") ]
define double @f10(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !"qqq") ]
  ret double %ftrunc
}

; Test multiple fp.except bundles.
; CHECK-NEXT: Multiple "fp.except" operand bundles
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.except"(metadata !"strict"), "fp.except"(metadata !"strict") ]
define double @f11(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.except"(metadata !"strict"), "fp.except"(metadata !"strict") ]
  ret double %ftrunc
}

; Test fp.except bundle that has more than one operands.
; CHECK-NEXT: Expected exactly one "fp.except" bundle operand
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.except"(metadata !"strict", metadata !"strict") ]
define double @f12(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.except"(metadata !"strict", metadata !"strict") ]
  ret double %ftrunc
}

; Test fp.except bundle that has non-metadata operand.
; CHECK-NEXT: Value of a "fp.except" bundle operand must be a metadata
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.except"(i32 0) ]
define double @f13(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.except"(i32 0) ]
  ret double %ftrunc
}

; Test fp.except bundle that has non-string operand.
; CHECK-NEXT: Value of a "fp.except" bundle operand must be a string
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.except"(metadata i64 3) ]
define double @f14(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.except"(metadata !{i64 3}) ]
  ret double %ftrunc
}

; Test fp.except bundle that specifies incorrect value.
; CHECK-NEXT: Value of a "fp.except" bundle operand is not a correct exception behavior
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.except"(metadata !"qqq") ]
define double @f15(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.except"(metadata !"qqq") ]
  ret double %ftrunc
}

; Test fp.control bundle specifies two input demormal modes.
; CHECK-NEXT: Input denormal mode is specified more that once
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata !"denorm.in=ieee", metadata !"denorm.in=ieee") ]
define double @f16(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !"denorm.in=ieee", metadata !"denorm.in=ieee") ]
  ret double %ftrunc
}

; Test fp.control bundle specifies two output demormal modes.
; CHECK-NEXT: Output denormal mode is specified more that once
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata !"denorm.out=ieee", metadata !"denorm.out=ieee") ]
define double @f17(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !"denorm.out=ieee", metadata !"denorm.out=ieee") ]
  ret double %ftrunc
}

; Test fp.control bundle specifies invalid input demormal modes.
; CHECK-NEXT: Invalid input denormal mode
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata !"denorm.in=qqq") ]
define double @f18(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !"denorm.in=qqq") ]
  ret double %ftrunc
}

; Test fp.control bundle specifies invalid output demormal modes.
; CHECK-NEXT: Invalid output denormal mode
; CHECK-NEXT:   %ftrunc = call double @llvm.trunc.f64(double %a) #{{[0-9]+}} [ "fp.control"(metadata !"denorm.out=qqq") ]
define double @f19(double %a) #0 {
entry:
  %ftrunc = call double @llvm.trunc.f64(double %a) #0 [ "fp.control"(metadata !"denorm.out=qqq") ]
  ret double %ftrunc
}


attributes #0 = { strictfp }
