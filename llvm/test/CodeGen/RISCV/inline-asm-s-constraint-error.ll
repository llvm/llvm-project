; RUN: not llc -mtriple=riscv64 < %s 2>&1 | FileCheck %s

@a = external global [4 x i32], align 16

; CHECK-COUNT-2: error: invalid operand for inline asm constraint 's'
; CHECK-NOT:     error:
define void @test(i64 %i) {
entry:
  %x = alloca i32, align 4
  %ai = getelementptr inbounds [4 x i32], ptr @a, i64 0, i64 %i
  call void asm sideeffect "", "s,~{dirflag},~{fpsr},~{flags}"(ptr %x)
  call void asm sideeffect "", "s,~{dirflag},~{fpsr},~{flags}"(ptr %ai)
  ret void
}
