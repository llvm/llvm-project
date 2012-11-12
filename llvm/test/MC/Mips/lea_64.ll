; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 %s -o - \
; RUN:  | llvm-objdump -disassemble -triple mips64el - \
; RUN:  | FileCheck %s

@p = external global i32*

define void @f1() nounwind {
entry:
; CHECK: .text:
; CHECK-NOT: addiu {{[0-9,a-f]+}}, {{[0-9,a-f]+}}, {{[0-9]+}}

  %a = alloca [10 x i32], align 4
  %arraydecay = getelementptr inbounds [10 x i32]* %a, i64 0, i64 0
  store i32* %arraydecay, i32** @p, align 8
  ret void

; CHECK: jr $ra
}
