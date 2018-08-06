; Positive test for inline register constraints
;
; RUN: llc -no-integrated-as -march=mipsel -relocation-model=pic < %s | \
; RUN:     FileCheck -check-prefixes=ALL,LE32,GAS %s
; RUN: llc -no-integrated-as -march=mips -relocation-model=pic < %s | \
; RUN:     FileCheck -check-prefixes=ALL,BE32,GAS %s

; IAS might not print in the same way since it parses the assembly.
; RUN: llc -march=mipsel -relocation-model=pic < %s | \
; RUN:     FileCheck -check-prefixes=ALL,LE32,IAS %s
; RUN: llc -march=mips -relocation-model=pic < %s | \
; RUN:     FileCheck -check-prefixes=ALL,BE32,IAS %s

%union.u_tag = type { i64 }
%struct.anon = type { i32, i32 }
@uval = common global %union.u_tag zeroinitializer, align 8

; X with -3
define i32 @constraint_X() nounwind {
entry:
; ALL-LABEL: constraint_X:
; ALL:           #APP
; GAS:           addiu ${{[0-9]+}}, ${{[0-9]+}}, 0xfffffffffffffffd
; IAS:           addiu ${{[0-9]+}}, ${{[0-9]+}}, -3
; ALL:           #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, ${2:X}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; x with -3
define i32 @constraint_x() nounwind {
entry:
; ALL-LABEL: constraint_x:
; ALL: #APP
; GAS: addiu ${{[0-9]+}}, ${{[0-9]+}}, 0xfffd
; This is _also_ -3 because uimm16 values are silently coerced to simm16 when
; it would otherwise fail to match.
; IAS: addiu ${{[0-9]+}}, ${{[0-9]+}}, -3
; ALL: #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, ${2:x}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; d with -3
define i32 @constraint_d() nounwind {
entry:
; ALL-LABEL: constraint_d:
; ALL:   #APP
; ALL:   addiu ${{[0-9]+}}, ${{[0-9]+}}, -3
; ALL:   #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, ${2:d}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; m with -3
define i32 @constraint_m() nounwind {
entry:
; ALL-LABEL: constraint_m:
; ALL:   #APP
; ALL:   addiu ${{[0-9]+}}, ${{[0-9]+}}, -4
; ALL:   #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, ${2:m}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; y with 4
define i32 @constraint_y_4() nounwind {
entry:
; ALL-LABEL: constraint_y_4:
; ALL:   #APP
; ALL:   addiu ${{[0-9]+}}, ${{[0-9]+}}, 2
; ALL:   #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, ${2:y}", "=r,r,I"(i32 7, i32 4) ;
  ret i32 0
}

; z with -3
define void @constraint_z_0() nounwind {
entry:
; ALL-LABEL: constraint_z_0:
; ALL:    #APP
; ALL:    addiu ${{[0-9]+}}, ${{[0-9]+}}, -3
; ALL:    #NO_APP
  tail call i32 asm sideeffect "addiu $0, $1, ${2:z}", "=r,r,I"(i32 7, i32 -3) ;
  ret void
}

; z with 0
define void @constraint_z_1() nounwind {
entry:
; ALL-LABEL: constraint_z_1:
; ALL:    #APP
; GAS:    addu ${{[0-9]+}}, ${{[0-9]+}}, $0
; IAS:    move ${{[0-9]+}}, ${{[0-9]+}}
; ALL:    #NO_APP
  tail call i32 asm sideeffect "addu $0, $1, ${2:z}", "=r,r,I"(i32 7, i32 0) nounwind
  ret void
}

; z with non-zero and the "r"(register) and "J"(integer zero) constraints
define void @constraint_z_2() nounwind {
entry:
; ALL-LABEL: constraint_z_2:
; ALL:    #APP
; ALL:    mtc0 ${{[1-9][0-9]?}}, ${{[0-9]+}}
; ALL:    #NO_APP
  call void asm sideeffect "mtc0 ${0:z}, $$12", "Jr"(i32 7) nounwind
  ret void
}

; z with zero and the "r"(register) and "J"(integer zero) constraints
define void @constraint_z_3() nounwind {
entry:
; ALL-LABEL: constraint_z_3:
; ALL:    #APP
; GAS:    mtc0 $0, ${{[0-9]+}}
; IAS:    mtc0 $zero, ${{[0-9]+}}, 0
; ALL:    #NO_APP
  call void asm sideeffect "mtc0 ${0:z}, $$12", "Jr"(i32 0) nounwind
  ret void
}

; z with non-zero and just the "r"(register) constraint
define void @constraint_z_4() nounwind {
entry:
; ALL-LABEL: constraint_z_4:
; ALL:    #APP
; ALL:    mtc0 ${{[1-9][0-9]?}}, ${{[0-9]+}}
; ALL:    #NO_APP
  call void asm sideeffect "mtc0 ${0:z}, $$12", "r"(i32 7) nounwind
  ret void
}

; z with zero and just the "r"(register) constraint
define void @constraint_z_5() nounwind {
entry:
; ALL-LABEL: constraint_z_5:
; FIXME: Check for $0, instead of other registers.
;        We should be using $0 directly in this case, not real registers.
;        When the materialization of 0 gets fixed, this test will fail.
; ALL:    #APP
; ALL:    mtc0 ${{[1-9][0-9]?}}, ${{[0-9]+}}
; ALL:    #NO_APP
  call void asm sideeffect "mtc0 ${0:z}, $$12", "r"(i32 0) nounwind
  ret void
}

; A long long in 32 bit mode (use to assert)
define i32 @constraint_longlong() nounwind {
entry:
; ALL-LABEL: constraint_longlong:
; ALL:           #APP
; ALL:           addiu ${{[0-9]+}}, ${{[0-9]+}}, 3
; ALL:           #NO_APP
  tail call i64 asm sideeffect "addiu $0, $1, $2 \0A\09", "=r,r,X"(i64 1229801703532086340, i64 3) nounwind
  ret i32 0
}

; In little endian the source reg will be 4 bytes into the long long
; In big endian the source reg will also be 4 bytes into the long long
define i32 @constraint_D() nounwind {
entry:
; ALL-LABEL: constraint_D:
; ALL:           lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
; ALL:           lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
; ALL:           lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
; ALL:           #APP
; LE32:          or ${{[0-9]+}}, $[[SECOND]], ${{[0-9]+}}
; BE32:          or ${{[0-9]+}}, $[[SECOND]], ${{[0-9]+}}
; ALL:           #NO_APP
  %bosco = load i64, i64* getelementptr inbounds (%union.u_tag, %union.u_tag* @uval, i32 0, i32 0), align 8
  %trunc1 = trunc i64 %bosco to i32
  tail call i32 asm sideeffect "or $0, ${1:D}, $2", "=r,r,r"(i64 %bosco, i32 %trunc1) nounwind
  ret i32 0
}

; In little endian the source reg will be 0 bytes into the long long
; In big endian the source reg will be 4 bytes into the long long
define i32 @constraint_L() nounwind {
entry:
; ALL-LABEL: constraint_L:
; ALL:           lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
; ALL:           lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
; ALL:           lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
; ALL:           #APP
; LE32:          or ${{[0-9]+}}, $[[FIRST]], ${{[0-9]+}}
; BE32:          or ${{[0-9]+}}, $[[SECOND]], ${{[0-9]+}}
; ALL:           #NO_APP
  %bosco = load i64, i64* getelementptr inbounds (%union.u_tag, %union.u_tag* @uval, i32 0, i32 0), align 8
  %trunc1 = trunc i64 %bosco to i32
  tail call i32 asm sideeffect "or $0, ${1:L}, $2", "=r,r,r"(i64 %bosco, i32 %trunc1) nounwind
  ret i32 0
}

; In little endian the source reg will be 4 bytes into the long long
; In big endian the source reg will be 0 bytes into the long long
define i32 @constraint_M() nounwind {
entry:
; ALL-LABEL: constraint_M:
; ALL:           lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
; ALL:           lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
; ALL:           lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
; ALL:           #APP
; LE32:          or ${{[0-9]+}}, $[[SECOND]], ${{[0-9]+}}
; BE32:          or ${{[0-9]+}}, $[[FIRST]], ${{[0-9]+}}
; ALL:           #NO_APP
  %bosco = load i64, i64* getelementptr inbounds (%union.u_tag, %union.u_tag* @uval, i32 0, i32 0), align 8
  %trunc1 = trunc i64 %bosco to i32
  tail call i32 asm sideeffect "or $0, ${1:M}, $2", "=r,r,r"(i64 %bosco, i32 %trunc1) nounwind
  ret i32 0
}
