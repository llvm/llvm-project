; RUN: llc < %s -mtriple=s390x-linux-gnu \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=soft-float \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SOFT-FLOAT
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=-soft-float \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NO-SOFT-FL
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=-vector \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NO-VECTOR
;
; Test per function attributes and command line arguments that override them.

attributes #1 = { "target-cpu"="z14" "target-features"="+vector" "use-soft-float"="false" }
define double @fun1(double* %A) #1 {
; CHECK-LABEL: fun1:
; DEFAULT:     ld %f0, 0(%r2)
; SOFT-FLOAT:  lg %r2, 0(%r2)
; NO-SOFT-FL:  ld %f0, 0(%r2)
; NO-VECTOR:   ld %f0, 0(%r2)
; CHECK-NEXT:  br %r14
entry:
  %0 = load double, double* %A
  ret double %0
}

attributes #2 = { "target-cpu"="z14" "target-features"="+vector" "use-soft-float"="true" }
define double @fun2(double* %A) #2 {
; CHECK-LABEL: fun2:
; DEFAULT:     lg %r2, 0(%r2)
; SOFT-FLOAT:  lg %r2, 0(%r2)
; NO-SOFT-FL:  lg %r2, 0(%r2)
; NO-VECTOR:   lg %r2, 0(%r2)
; CHECK-NEXT:  br %r14
entry:
  %0 = load double, double* %A
  ret double %0
}

attributes #3 = { "target-cpu"="z14" "target-features"="+vector" "use-soft-float"="false" }
define <2 x double> @fun3(<2 x double>* %A) #3 {
; CHECK-LABEL:     fun3:
; DEFAULT:         vl %v24, 0(%r2), 4
; SOFT-FLOAT:      lg %r0, 0(%r2)
; SOFT-FLOAT-NEXT: lg %r3, 8(%r2)
; SOFT-FLOAT-NEXT: lgr %r2, %r0
; NO-SOFT-FL:      vl %v24, 0(%r2), 4
; NO-VECTOR:       ld %f0, 0(%r2)
; NO-VECTOR-NEXT:  ld %f2, 8(%r2)
; CHECK-NEXT:      br %r14
entry:
  %0 = load <2 x double>, <2 x double>* %A
  ret <2 x double> %0
}

attributes #4 = { "target-cpu"="z14" "target-features"="+vector" "use-soft-float"="true" }
define <2 x double> @fun4(<2 x double>* %A) #4 {
; CHECK-LABEL:     fun4:
; DEFAULT:         lg %r0, 0(%r2)
; DEFAULT-NEXT:    lg %r3, 8(%r2)
; DEFAULT-NEXT:    lgr %r2, %r0
; SOFT-FLOAT:      lg %r0, 0(%r2)
; SOFT-FLOAT-NEXT: lg %r3, 8(%r2)
; SOFT-FLOAT-NEXT: lgr %r2, %r0
; NO-SOFT-FL:      lg %r0, 0(%r2)
; NO-SOFT-FL-NEXT: lg %r3, 8(%r2)
; NO-SOFT-FL-NEXT: lgr %r2, %r0
; NO-VECTOR:       lg %r0, 0(%r2)
; NO-VECTOR-NEXT:  lg %r3, 8(%r2)
; NO-VECTOR-NEXT:  lgr %r2, %r0
; CHECK-NEXT:      br %r14
entry:
  %0 = load <2 x double>, <2 x double>* %A
  ret <2 x double> %0
}

attributes #5 = { "target-cpu"="z14" "target-features"="-vector" "use-soft-float"="false" }
define <2 x double> @fun5(<2 x double>* %A) #5 {
; CHECK-LABEL:     fun5:
; DEFAULT:         ld %f0, 0(%r2)
; DEFAULT-NEXT:    ld %f2, 8(%r2)
; SOFT-FLOAT:      lg %r0, 0(%r2)
; SOFT-FLOAT-NEXT: lg %r3, 8(%r2)
; SOFT-FLOAT-NEXT: lgr %r2, %r0
; NO-SOFT-FL:      ld %f0, 0(%r2)
; NO-SOFT-FL-NEXT: ld %f2, 8(%r2)
; NO-VECTOR:       ld %f0, 0(%r2)
; NO-VECTOR-NEXT:  ld %f2, 8(%r2)
; CHECK-NEXT:      br %r14
entry:
  %0 = load <2 x double>, <2 x double>* %A
  ret <2 x double> %0
}

attributes #6 = { "target-cpu"="zEC12" "use-soft-float"="false" }
define <2 x double> @fun6(<2 x double>* %A) #6 {
; CHECK-LABEL:     fun6:
; DEFAULT:         ld %f0, 0(%r2)
; DEFAULT-NEXT:    ld %f2, 8(%r2)
; SOFT-FLOAT:      lg %r0, 0(%r2)
; SOFT-FLOAT-NEXT: lg %r3, 8(%r2)
; SOFT-FLOAT-NEXT: lgr %r2, %r0
; NO-SOFT-FL:      ld %f0, 0(%r2)
; NO-SOFT-FL-NEXT: ld %f2, 8(%r2)
; NO-VECTOR:       ld %f0, 0(%r2)
; NO-VECTOR-NEXT:  ld %f2, 8(%r2)
; CHECK-NEXT:      br %r14
entry:
  %0 = load <2 x double>, <2 x double>* %A
  ret <2 x double> %0
}

attributes #7 = { "target-cpu"="zEC12" "target-features"="+vector" "use-soft-float"="false" }
define <2 x double> @fun7(<2 x double>* %A) #7 {
; CHECK-LABEL:     fun7:
; DEFAULT:         vl %v24, 0(%r2), 4
; SOFT-FLOAT:      lg %r0, 0(%r2)
; SOFT-FLOAT-NEXT: lg %r3, 8(%r2)
; SOFT-FLOAT-NEXT: lgr %r2, %r0
; NO-SOFT-FL:      vl %v24, 0(%r2), 4
; NO-VECTOR:       ld %f0, 0(%r2)
; NO-VECTOR-NEXT:  ld %f2, 8(%r2)
; CHECK-NEXT:      br %r14
entry:
  %0 = load <2 x double>, <2 x double>* %A
  ret <2 x double> %0
}
