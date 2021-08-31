; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

%struct = type { [4 x i32], [20 x i8] }

declare dso_local i32 @foo1(%struct* byval(%struct) %var)
declare dso_local void @foo(%struct* %var)

; for the byval parameter %x, make sure the memory for local variable and
; for parameter save area are not overlap.
; For the below case,
; the local variable space is r1 + 104 ~ r1 + 140
; the parameter save area is r1 + 32 ~ r1 + 68

define dso_local i32 @bar() {
; CHECK-LABEL: bar:
; CHECK:    addi 30, 1, 104
; CHECK:    li 3, 16
; CHECK:    lxvd2x 0, 30, 3
; CHECK:    li 3, 48
; CHECK:    stxvd2x 0, 1, 3
; CHECK:    li 3, 32
; CHECK:    lxvd2x 0, 0, 30
; CHECK:    stxvd2x 0, 1, 3
; CHECK:    lwz 3, 136(1)
; CHECK:    stw 3, 64(1)
entry:
  %x = alloca %struct, align 4
  call void @foo(%struct* %x)
  %r = call i32 @foo1(%struct* byval(%struct) %x)
  ret i32 %r
}
