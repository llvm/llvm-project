; RUN: llc -march=mipsel -mcpu=mips32 -relocation-model=pic \
; RUN:     -fast-isel=true -fast-isel-abort=3 < %s | FileCheck %s
; RUN: llc -march=mipsel -mcpu=mips32r2 -relocation-model=pic \
; RUN:     -fast-isel=true -fast-isel-abort=3 < %s | FileCheck %s

@ARR = external global [10 x i32], align 4

define void @foo() {
; CHECK-LABEL: foo

; CHECK-DAG:    lw      $[[ARR:[0-9]+]], %got(ARR)({{.*}})
; CHECK-DAG:    addiu   $[[T0:[0-9]+]], $zero, 12345
; CHECK:        sw      $[[T0]], 8($[[ARR]])

entry:
  store i32 12345, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @ARR, i32 0, i32 2), align 4
  ret void
}
