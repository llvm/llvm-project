; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec -xcoff-traceback-table=false \
; RUN:  -vec-extabi -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=ASM32,ASM %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec -xcoff-traceback-table=false \
; RUN:  -vec-extabi -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=ASM64,ASM %s

define dso_local <4 x i32> @vec_callee(<4 x i32> %vec1, <4 x i32> %vec2, <4 x i32> %vec3, <4 x i32> %vec4, <4 x i32> %vec5, <4 x i32> %vec6, <4 x i32> %vec7, <4 x i32> %vec8, <4 x i32> %vec9, <4 x i32> %vec10, <4 x i32> %vec11, <4 x i32> %vec12) {
entry:
  %add = add <4 x i32> %vec1, %vec2
  %add1 = add <4 x i32> %add, %vec3
  %add2 = add <4 x i32> %add1, %vec4
  %add3 = add <4 x i32> %add2, %vec5
  %add4 = add <4 x i32> %add3, %vec6
  %add5 = add <4 x i32> %add4, %vec7
  %add6 = add <4 x i32> %add5, %vec8
  %add7 = add <4 x i32> %add6, %vec9
  %add8 = add <4 x i32> %add7, %vec10
  %add9 = add <4 x i32> %add8, %vec11
  %add10 = add <4 x i32> %add9, %vec12
  ret <4 x i32> %add10
}

; ASM-LABEL:     .vec_callee:

; ASM:           %entry
; ASM-DAG:       vadduwm 2, 2, 3
; ASM-DAG:       vadduwm 2, 2, 4
; ASM-DAG:       vadduwm 2, 2, 5
; ASM-DAG:       vadduwm 2, 2, 6
; ASM-DAG:       vadduwm 2, 2, 7
; ASM-DAG:       vadduwm 2, 2, 8
; ASM-DAG:       vadduwm 2, 2, 9
; ASM-DAG:       vadduwm 2, 2, 10
; ASM-DAG:       vadduwm 2, 2, 11
; ASM-DAG:       vadduwm 2, 2, 12
; ASM-DAG:       vadduwm 2, 2, 13
; ASM:           blr

define dso_local i32 @vec_caller() {
entry:
  %call = call <4 x i32> @vec_callee(<4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> <i32 5, i32 6, i32 7, i32 8>, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, <4 x i32> <i32 13, i32 14, i32 15, i32 16>, <4 x i32> <i32 17, i32 18, i32 19, i32 20>, <4 x i32> <i32 21, i32 22, i32 23, i32 24>, <4 x i32> <i32 25, i32 26, i32 27, i32 28>, <4 x i32> <i32 29, i32 30, i32 31, i32 32>, <4 x i32> <i32 33, i32 34, i32 35, i32 36>, <4 x i32> <i32 37, i32 38, i32 39, i32 40>, <4 x i32> <i32 41, i32 42, i32 43, i32 44>, <4 x i32> <i32 45, i32 46, i32 47, i32 48>)
  ret i32 0
}

; ASM-LABEL:     .vec_caller:
; ASM32:         # %bb.0:                                # %entry
; ASM32-DAG:     mflr 0
; ASM32-DAG:     stw 0, 8(1)
; ASM32-DAG:     stwu 1, -64(1)
; ASM32-DAG:     lwz [[REG1:[0-9]+]], L..C0(2)
; ASM32-DAG:     lxvw4x 34, 0, [[REG1]]
; ASM32-DAG:     lwz [[REG2:[0-9]+]], L..C1(2)
; ASM32-DAG:     lxvw4x 35, 0, [[REG2]]
; ASM32-DAG:     lwz [[REG3:[0-9]+]], L..C2(2)
; ASM32-DAG:     lxvw4x 36, 0, [[REG3]]
; ASM32-DAG:     lwz [[REG4:[0-9]+]], L..C3(2)
; ASM32-DAG:     lxvw4x 37, 0, [[REG4]]
; ASM32-DAG:     lwz [[REG5:[0-9]+]], L..C4(2)
; ASM32-DAG:     lxvw4x 38, 0, [[REG5]]
; ASM32-DAG:     lwz [[REG6:[0-9]+]], L..C5(2)
; ASM32-DAG:     lxvw4x 39, 0, [[REG6]]
; ASM32-DAG:     lwz [[REG7:[0-9]+]], L..C6(2)
; ASM32-DAG:     lxvw4x 40, 0, [[REG7]]
; ASM32-DAG:     lwz [[REG8:[0-9]+]], L..C7(2)
; ASM32-DAG:     lxvw4x 41, 0, [[REG8]]
; ASM32-DAG:     lwz [[REG9:[0-9]+]], L..C8(2)
; ASM32-DAG:     lxvw4x 42, 0, [[REG9]]
; ASM32-DAG:     lwz [[REG10:[0-9]+]], L..C9(2)
; ASM32-DAG:     lxvw4x 43, 0, [[REG10]]
; ASM32-DAG:     lwz [[REG11:[0-9]+]], L..C10(2)
; ASM32-DAG:     lxvw4x 44, 0, [[REG11]]
; ASM32-DAG:     lwz [[REG12:[0-9]+]], L..C11(2)
; ASM32-DAG:     lxvw4x 45, 0, [[REG12]]
; ASM32-DAG:     bl .vec_callee
; ASM32-DAG:     li 3, 0
; ASM32-DAG:     addi 1, 1, 64
; ASM32-DAG:     lwz 0, 8(1)
; ASM32-DAG:     mtlr 0
; ASM32:         blr

; ASM64:         # %entry
; ASM64-DAG:     std 0, 16(1)
; ASM64-DAG:     stdu 1, -112(1)
; ASM64-DAG:     ld [[REG1:[0-9]+]], L..C0(2)
; ASM64-DAG:     lxvw4x 34, 0, [[REG1]]
; ASM64-DAG:     ld [[REG2:[0-9]+]], L..C1(2)
; ASM64-DAG:     lxvw4x 35, 0, [[REG2]]
; ASM64-DAG:     ld [[REG3:[0-9]+]], L..C2(2)
; ASM64-DAG:     lxvw4x 36, 0, [[REG3]]
; ASM64-DAG:     ld [[REG4:[0-9]+]], L..C3(2)
; ASM64-DAG:     lxvw4x 37, 0, [[REG4]]
; ASM64-DAG:     ld [[REG5:[0-9]+]], L..C4(2)
; ASM64-DAG:     lxvw4x 38, 0, [[REG5]]
; ASM64-DAG:     ld [[REG6:[0-9]+]], L..C5(2)
; ASM64-DAG:     lxvw4x 39, 0, [[REG6]]
; ASM64-DAG:     ld [[REG7:[0-9]+]], L..C6(2)
; ASM64-DAG:     lxvw4x 40, 0, [[REG7]]
; ASM64-DAG:     ld [[REG8:[0-9]+]], L..C7(2)
; ASM64-DAG:     lxvw4x 41, 0, [[REG8]]
; ASM64-DAG:     ld [[REG9:[0-9]+]], L..C8(2)
; ASM64-DAG:     lxvw4x 42, 0, [[REG9]]
; ASM64-DAG:     ld [[REG10:[0-9]+]], L..C9(2)
; ASM64-DAG:     lxvw4x 43, 0, [[REG10]]
; ASM64-DAG:     ld [[REG11:[0-9]+]], L..C10(2)
; ASM64-DAG:     lxvw4x 44, 0, [[REG11]]
; ASM64-DAG:     ld [[REG12:[0-9]+]], L..C11(2)
; ASM64-DAG:     lxvw4x 45, 0, [[REG12]]
; ASM64-DAG:     bl .vec_callee
; ASM64-DAG:     li 3, 0
; ASM64-DAG:     addi 1, 1, 112
; ASM64-DAG:     ld 0, 16(1)
; ASM64-DAG:     mtlr 0
; ASM64:         blr
