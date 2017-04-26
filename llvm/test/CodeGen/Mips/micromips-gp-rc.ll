; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
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

; CHECK: addu $gp, ${{[0-9]+}}

