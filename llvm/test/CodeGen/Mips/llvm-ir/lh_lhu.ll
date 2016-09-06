; RUN: llc < %s -march=mips -mcpu=mips32r2 -mattr=+micromips -relocation-model=pic | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips -relocation-model=pic | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips -relocation-model=pic | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips64r6 -target-abi n64 -mattr=+micromips -relocation-model=pic | FileCheck %s

@us = global i16 0, align 2

define i32 @lhfunc() {
entry:
; CHECK-LABEL: lhfunc
; CHECK: lh $[[REG1:[0-9]+]], 0(${{[0-9]+}})
  %0 = load i16, i16* @us, align 2
  %conv = sext i16 %0 to i32
  ret i32 %conv
}

define i16 @lhfunc_atomic() {
entry:
; CHECK-LABEL: lhfunc_atomic
; CHECK: lh $[[REG1:[0-9]+]], 0(${{[0-9]+}})
  %0 = load atomic i16, i16* @us acquire, align 2
  ret i16 %0
}

define i32 @lhufunc() {
entry:
; CHECK-LABEL: lhufunc
; CHECK: lhu $[[REG1:[0-9]+]], 0(${{[0-9]+}})
  %0 = load i16, i16* @us, align 2
  %conv = zext i16 %0 to i32
  ret i32 %conv
}
