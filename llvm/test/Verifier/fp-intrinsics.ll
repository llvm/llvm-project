; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)

; Test an illegal value for the rounding mode argument.
; CHECK: invalid rounding mode argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynomic", metadata !"fpexcept.strict") #{{[0-9]+}}
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
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.restrict") #{{[0-9]+}}
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
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynomic", metadata !"fpexcept.strict") #{{[0-9]+}}
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
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynamic", metadata !"fpexcept.restrict") #{{[0-9]+}}
define double @f5(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                        double %a,
                                        metadata !"round.dynamic",
                                        metadata !"fpexcept.restrict") #0
  ret double %fadd
}

; Test "fpe.except" operand bundle attached to the call that may not have such.
; CHECK-NEXT: This function call may not have fpe.except bundle
; CHECK-NEXT:   %add = call i32 @llvm.sadd.sat.i32(i32 %a, i32 %b) [ "fpe.except"(metadata !"strict") ]
define i32 @f6(i32 %a, i32 %b) {
entry:
  %add = call i32 @llvm.sadd.sat.i32(i32 %a, i32 %b) [ "fpe.except"(metadata !"strict") ]
  ret i32 %add
}

; Test "fpe.control" operand bundle attached to the call that may not have such.
; CHECK-NEXT: This function call may not have fpe.control bundle
; CHECK-NEXT:   %add = call i32 @llvm.sadd.sat.i32(i32 %a, i32 %b) [ "fpe.control"(metadata !"rte") ]
define i32 @f7(i32 %a, i32 %b) {
entry:
  %add = call i32 @llvm.sadd.sat.i32(i32 %a, i32 %b) [ "fpe.control"(metadata !"rte") ]
  ret i32 %add
}

attributes #0 = { strictfp }
