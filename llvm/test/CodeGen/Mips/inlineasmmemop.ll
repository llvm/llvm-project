; RUN: llc -march=mipsel < %s | FileCheck %s

@g1 = external global i32

define i32 @f1(i32 %x) nounwind {
entry:
; CHECK: addiu $[[T0:[0-9]+]], $sp
; CHECK: #APP
; CHECK: sw $4, 0($[[T0]])
; CHECK: #NO_APP
; CHECK: #APP
; CHECK: lw $[[T3:[0-9]+]], 0($[[T0]])
; CHECK: #NO_APP
; CHECK: lw  $[[T1:[0-9]+]], %got(g1)
; CHECK: sw  $[[T3]], 0($[[T1]])

  %l1 = alloca i32, align 4
  call void asm "sw $1, $0", "=*m,r"(i32* %l1, i32 %x) nounwind
  %0 = call i32 asm "lw $0, $1", "=r,*m"(i32* %l1) nounwind
  store i32 %0, i32* @g1, align 4
  ret i32 %0
}

