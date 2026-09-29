; RUN: llc -mtriple i686-gnu -filetype asm -o - %s 2>&1 | FileCheck %s

define void @test_L_ff() {
entry:
  call void asm "", "L,~{dirflag},~{fpsr},~{flags}"(i32 255)
  ret void
}

; CHECK-NOT: error: invalid operand for inline asm constraint 'L'

define void @test_L_ffff() {
entry:
  call void asm "", "L,~{dirflag},~{fpsr},~{flags}"(i32 65535)
  ret void
}

; CHECK-NOT: error: invalid operand for inline asm constraint 'L'

define void @test_M_1() {
entry:
  call void asm "", "M,~{dirflag},~{fpsr},~{flags}"(i32 1)
  ret void
}

; CHECK-NOT: error: invalid operand for inline asm constraint 'M'

define void @test_O_64() {
entry:
  call void asm "", "O,~{dirflag},~{fpsr},~{flags}"(i32 64)
  ret void
}

; CHECK-NOT: error: invalid operand for inline asm constraint 'O'

