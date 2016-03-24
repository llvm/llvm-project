; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s
; For microMIPS64, also check 32 to 64 bit registers and 64 to 32 bit register copies
; RUN: llc -march=mips -mcpu=mips64r6 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

@g = external global i32

; Function Attrs: noreturn nounwind
define void @foo() #0 {
entry:
  %0 = load i32, i32* @g, align 4
  tail call void @exit(i32 signext %0)
  unreachable
}

; Function Attrs: noreturn
declare void @exit(i32 signext)

; CHECK: move $gp, ${{[0-9]+}}

