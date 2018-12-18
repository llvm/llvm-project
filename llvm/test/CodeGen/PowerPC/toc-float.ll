; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 <%s | FileCheck -check-prefix=CHECK-P9 %s
; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 <%s | FileCheck -check-prefix=CHECK-P8 %s

; As the constant could be represented as float, a float is
; loaded from constant pool.
define double @doubleConstant1() {
  ret double 1.400000e+01

; CHECK-P9-LABEL: doubleConstant1:
; CHECK-P9: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P9: lfs {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
; CHECK-P8-LABEL: doubleConstant1:
; CHECK-P8: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P8: lfs {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
}

; As the constant couldn't be represented as float, a double is
; loaded from constant pool.
define double @doubleConstant2() {
  ret double 2.408904e+01

; CHECK-P9-LABEL: doubleConstant2:
; CHECK-P9: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P9: lfd {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
; CHECK-P8-LABEL: doubleConstant2:
; CHECK-P8: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P8: lfd {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
}

@FArr = hidden local_unnamed_addr global [10 x float] zeroinitializer, align 4

define float @floatConstantArray() local_unnamed_addr  {
  %1 = load float, float* getelementptr inbounds ([10 x float], [10 x float]* @FArr, i64 0, i64 3), align 4
  %2 = fadd float %1, 0x400B333340000000
  ret float %2

; CHECK-P9-LABEL: floatConstantArray 
; CHECK-P9: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha+[[REG2:[0-9]+]]
; CHECK-P9: lfs {{[0-9]+}}, [[VAR]]@toc@l+[[REG2]]([[REG1]])
; CHECK-P8-LABEL: floatConstantArray 
; CHECK-P8: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P8: addi [[REG2:[0-9]+]], [[REG1]], [[VAR]]@toc@l
; CHECK-P8: lfs {{[0-9]+}}, 12([[REG2]])
}

define float @floatConstant() {
  ret float 0x400470A3E0000000

; CHECK-P9-LABEL: floatConstant:
; CHECK-P9: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P9: lfs {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
; CHECK-P8-LABEL: floatConstant:
; CHECK-P8: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P8: lfs {{[0-9]+}}, [[VAR]]@toc@l([[REG1]])
}

; llvm put the hidden globals into the TOC table.
; TODO - do some analysis and decide which globals could be put into TOC.
@d = hidden local_unnamed_addr global [200 x double] zeroinitializer, align 8

define double @doubleConstantArray()  {
  %1 = load double, double* getelementptr inbounds ([200 x double], [200 x double]* @d, i64 0, i64 3), align 8
  %2 = fadd double %1, 6.880000e+00
  ret double %2

; CHECK-P9-LABEL: doubleConstantArray
; CHECK-P9: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha+[[REG2:[0-9]+]]
; CHECK-P9: lfd {{[0-9]+}}, [[VAR]]@toc@l+[[REG2]]([[REG1]])
; CHECK-P8-LABEL: doubleConstantArray
; CHECK-P8: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P8: addi [[REG2:[0-9]+]], [[REG1]], [[VAR]]@toc@l
; CHECK-P8: lfd {{[0-9]+}}, 24([[REG2]])
}

@arr = hidden local_unnamed_addr global [20000 x double] zeroinitializer, align 8

define double @doubleLargeConstantArray()  {
  %1 = load double, double* getelementptr inbounds ([20000 x double], [20000 x double]* @arr, i64 0, i64 4096), align 8
  %2 = fadd double %1, 6.880000e+00
  ret double %2

; Access an element with an offset that doesn't fit in the displacement field of LFD. 
; CHECK-P9-LABEL: doubleLargeConstantArray
; CHECK-P9: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P9: li [[REG2:[0-9]+]], 0 
; CHECK-P9: addi [[REG3:[0-9]+]], [[REG1]], [[VAR:[a-z0-9A-Z_.]+]]@toc@l
; CHECK-P9: ori [[REG4:[0-9]+]], [[REG2]], 32768 
; CHECK-P9: lfdx {{[0-9]+}}, [[REG3]], [[REG4]] 
; CHECK-P8-LABEL: doubleLargeConstantArray
; CHECK-P8: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P8: li [[REG2:[0-9]+]], 0 
; CHECK-P8: addi [[REG3:[0-9]+]], [[REG1]], [[VAR:[a-z0-9A-Z_.]+]]@toc@l
; CHECK-P8: ori [[REG4:[0-9]+]], [[REG2]], 32768 
; CHECK-P8: lfdx {{[0-9]+}}, [[REG3]], [[REG4]] 
}

@vec_arr = global [10 x <4 x i32>] zeroinitializer, align 16

define <4 x i32> @vectorArray() #0 {
entry:
  %0 = load <4 x i32>, <4 x i32>* getelementptr inbounds ([10 x <4 x i32>], [10 x <4 x i32>]* @vec_arr, i64 0, i64 2), align 16
  ret <4 x i32> %0

; CHECK-P9-LABEL: vectorArray
; CHECK-P9: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P9: ld [[REG2:[0-9]+]], [[VAR]]@toc@l([[REG1]])
; CHECK-P9: lxv {{[0-9]+}}, 32([[REG2]])
; CHECK-P8-LABEL: vectorArray
; CHECK-P8: addis [[REG1:[0-9]+]], 2, [[VAR:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-P8: ld [[REG2:[0-9]+]], [[VAR]]@toc@l([[REG1]])
; CHECK-P8: addi [[REG3:[0-9]+]], [[REG2]], 32
; CHECK-P8: lvx {{[0-9]+}}, 0, [[REG3]]
}
