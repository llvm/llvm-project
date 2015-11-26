; RUN: llc -no-integrated-as -march=mipsel < %s | \
; RUN:     FileCheck %s -check-prefix=ALL -check-prefix=GAS

define void @constraint_I() nounwind {
; First I with short
; ALL-LABEL: constraint_I:
; ALL:           #APP
; ALL:           addiu ${{[0-9]+}}, ${{[0-9]+}}, 4096
; ALL:           #NO_APP
  tail call i16 asm sideeffect "addiu $0, $1, $2", "=r,r,I"(i16 7, i16 4096) nounwind

; Then I with int
; ALL: #APP
; ALL: addiu ${{[0-9]+}}, ${{[0-9]+}}, -3
; ALL: #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, $2", "=r,r,I"(i32 7, i32 -3) nounwind
  ret void
}

define void @constraint_J() nounwind {
; Now J with 0
; ALL-LABEL: constraint_J:
; ALL: #APP
; ALL: addiu ${{[0-9]+}}, ${{[0-9]+}}, 0
; ALL: #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, $2\0A\09 ", "=r,r,J"(i32 7, i16 0) nounwind
  ret void
}

define void @constraint_K() nounwind {
; Now K with 64
; ALL: #APP
; GAS: addu ${{[0-9]+}}, ${{[0-9]+}}, 64
; ALL: #NO_APP	
  tail call i16 asm sideeffect "addu $0, $1, $2\0A\09 ", "=r,r,K"(i16 7, i16 64) nounwind
  ret void
}

define void @constraint_L() nounwind {
; Now L with 0x00100000
; ALL: #APP
; ALL: add ${{[0-9]+}}, ${{[0-9]+}}, ${{[0-9]+}}
; ALL: #NO_APP	
  tail call i32 asm sideeffect "add $0, $1, $3\0A\09", "=r,r,L,r"(i32 7, i32 1048576, i32 0) nounwind
  ret void
}

define void @constraint_N() nounwind {
; Now N with -3
; ALL: #APP
; ALL: addiu ${{[0-9]+}}, ${{[0-9]+}}, -3
; ALL: #NO_APP	
  tail call i32 asm sideeffect "addiu $0, $1, $2", "=r,r,N"(i32 7, i32 -3) nounwind
  ret void
}

define void @constraint_O() nounwind {
; Now O with -3
; ALL: #APP
; ALL: addiu ${{[0-9]+}}, ${{[0-9]+}}, -3
; ALL: #NO_APP	
  tail call i32 asm sideeffect "addiu $0, $1, $2", "=r,r,O"(i32 7, i16 -3) nounwind
  ret void
}

define void @constraint_P() nounwind {
; Now P with 65535
; ALL: #APP
; GAS: addiu ${{[0-9]+}}, ${{[0-9]+}}, 65535
; ALL: #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, $2", "=r,r,P"(i32 7, i32 65535) nounwind
  ret void
}
