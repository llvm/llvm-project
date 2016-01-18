; RUN: llc -march=mipsel < %s | FileCheck %s

; Simple memory
@g1 = external global i32

define i32 @f1(i32 %x) nounwind {
entry:
; CHECK-LABEL: f1:
; CHECK: #APP
; CHECK: sw $4, [[OFFSET:[0-9]+]]($sp)
; CHECK: #NO_APP
; CHECK: lw  $[[T1:[0-9]+]], %got(g1)
; CHECK: #APP
; CHECK: lw $[[T3:[0-9]+]], [[OFFSET]]($sp)
; CHECK: #NO_APP
; CHECK: sw  $[[T3]], 0($[[T1]])

  %l1 = alloca i32, align 4
  call void asm "sw $1, $0", "=*m,r"(i32* %l1, i32 %x) nounwind
  %0 = call i32 asm "lw $0, $1", "=r,*m"(i32* %l1) nounwind
  store i32 %0, i32* @g1, align 4
  ret i32 %0
}

; CHECK-LABEL: main:
; "D": Second word of a double word. This works for any memory element
; double or single.
; CHECK: #APP
; CHECK: lw ${{[0-9]+}}, 16(${{[0-9]+}});
; CHECK: #NO_APP

; No "D": First word of a double word. This works for any memory element
; double or single.
; CHECK: #APP
; CHECK: lw ${{[0-9]+}}, 12(${{[0-9]+}});
; CHECK: #NO_APP

@b = common global [20 x i32] zeroinitializer, align 4

define void @main() {
entry:
; Second word:
  tail call void asm sideeffect "    lw    $0, ${1:D};", "r,*m,~{$11}"(i32 undef, i32* getelementptr inbounds ([20 x i32], [20 x i32]* @b, i32 0, i32 3))
; First word. Notice, no 'D':
  tail call void asm sideeffect "    lw    $0, ${1};", "r,*m,~{$11}"(i32 undef, i32* getelementptr inbounds ([20 x i32], [20 x i32]* @b, i32 0, i32 3))
  ret void
}
