; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; CHECK: __hexagon_adddf3
; CHECK: __hexagon_subdf3

define void @foo(double* %acc, double %num, double %num2) nounwind {
entry:
  %acc.addr = alloca double*, align 4
  %num.addr = alloca double, align 8
  %num2.addr = alloca double, align 8
  store double* %acc, double** %acc.addr, align 4
  store double %num, double* %num.addr, align 8
  store double %num2, double* %num2.addr, align 8
  %0 = load double** %acc.addr, align 4
  %1 = load double* %0
  %2 = load double* %num.addr, align 8
  %add = fadd double %1, %2
  %3 = load double* %num2.addr, align 8
  %sub = fsub double %add, %3
  %4 = load double** %acc.addr, align 4
  store double %sub, double* %4
  ret void
}
