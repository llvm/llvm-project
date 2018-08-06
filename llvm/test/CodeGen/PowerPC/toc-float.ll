; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 <%s | FileCheck %s

; As the constant could be represented as float, a float is
; loaded from constant pool.
define double @doubleConstant1() {
  ret double 1.400000e+01
}

; CHECK-LABEL: doubleConstant1:
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK: lfs {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])

; As the constant couldn't be represented as float, a double is
; loaded from constant pool.
define double @doubleConstant2() {
  ret double 2.408904e+01
}

; CHECK-LABEL: doubleConstant2:
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK: lfd {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])

@FArr = hidden local_unnamed_addr global [10 x float] zeroinitializer, align 4

define float @floatConstantArray() local_unnamed_addr  {
  %1 = load float, float* getelementptr inbounds ([10 x float], [10 x float]* @FArr, i64 0, i64 3), align 4
  %2 = fadd float %1, 0x400B333340000000
  ret float %2
}

; CHECK-LABEL: floatConstantArray 
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha+[[REG2:[0-9]+]]
; CHECK: lfs {{[0-9]+}}, [[VAR]]@toc@l+[[REG2]]([[REG1]])

define float @floatConstant() {
  ret float 0x400470A3E0000000
}

; CHECK-LABEL: floatConstant:
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK: lfs {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])

; llvm put the hidden globals into the TOC table.
; TODO - do some analysis and decide which globals could be put into TOC.
@d = hidden local_unnamed_addr global [200 x double] zeroinitializer, align 8

define double @doubleConstantArray()  {
  %1 = load double, double* getelementptr inbounds ([200 x double], [200 x double]* @d, i64 0, i64 3), align 8
  %2 = fadd double %1, 6.880000e+00
  ret double %2
}

; CHECK-LABEL: doubleConstantArray
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha+[[REG2:[0-9]+]]
; CHECK: lfd {{[0-9]+}}, [[VAR]]@toc@l+[[REG2]]([[REG1]])

@arr = hidden local_unnamed_addr global [20000 x double] zeroinitializer, align 8

define double @doubleLargeConstantArray()  {
  %1 = load double, double* getelementptr inbounds ([20000 x double], [20000 x double]* @arr, i64 0, i64 4096), align 8
  %2 = fadd double %1, 6.880000e+00
  ret double %2
}

; access element that out of range
; CHECK-LABEL: doubleLargeConstantArray
; CHECK: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK: li [[REG2:[0-9]+]], 0 
; CHECK: addi [[REG3:[0-9]+]], [[REG1]], [[VAR:[a-z0-9A-Z_.]+]]@toc@l
; CHECK: ori [[REG4:[0-9]+]], [[REG2]], 32768 
; CHECK: lfdx {{[0-9]+}}, [[REG3]], [[REG4]] 
