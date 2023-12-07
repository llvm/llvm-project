; Positive test for inline register constraints
;
; RUN: llc -march=mipsel -mcpu=mips32 < %s | FileCheck %s
; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | FileCheck %s

define i32 @main() nounwind {
entry:

; r with char
;CHECK: #APP
;CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 23
;CHECK: #NO_APP
  tail call i8 asm sideeffect "addiu $0, $1, $2", "=r,r,n"(i8 27, i8 23) nounwind

; r with short
;CHECK: #APP
;CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 13
;CHECK: #NO_APP
  tail call i16 asm sideeffect "addiu $0, $1, $2", "=r,r,n"(i16 17, i16 13) nounwind

; r with int
;CHECK: #APP
;CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 3
;CHECK: #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, $2", "=r,r,n"(i32 7, i32 3) nounwind

; Now c with 1024: make sure register $25 is picked
; CHECK: #APP
; CHECK: addiu $25, ${{[0-9]+}}, 1024
; CHECK: #NO_APP
   tail call i32 asm sideeffect "addiu $0, $1, $2", "=c,c,I"(i32 4194304, i32 1024) nounwind

; Now l with 1024: make sure register lo is picked. We do this by checking the instruction
; after the inline expression for a mflo to pull the value out of lo.
; CHECK:       #APP
; CHECK:       mtlo ${{[0-9]+}}
; CHECK-NEXT:  madd ${{[0-9]+}}, ${{[0-9]+}}
; CHECK:       #NO_APP
; CHECK-NEXT:  mflo ${{[0-9]+}}
  %bosco = alloca i32, align 4
  call i32 asm sideeffect "\09mtlo $3 \0A\09\09madd $1, $2 ", "=l,r,r,r"(i32 7, i32 6, i32 44) nounwind
  store volatile i32 %4, ptr %bosco, align 4
 
; Check the 'l' constraint for 16-bit type.
; CHECK:       #APP
; CHECK:       mtlo ${{[0-9]+}}
; CHECK-NEXT:  madd ${{[0-9]+}}, ${{[0-9]+}}
; CHECK:       #NO_APP
; CHECK-NEXT:  mflo ${{[0-9]+}}
  %bosco16 = alloca i16, align 4
  call i16 asm sideeffect "\09mtlo $3 \0A\09\09madd $1, $2 ", "=l,r,r,r"(i32 7, i32 6, i32 44) nounwind
  store volatile i16 %5, ptr %bosco16, align 4

  ret i32 0
}
