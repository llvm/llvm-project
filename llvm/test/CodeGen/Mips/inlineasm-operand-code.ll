; Positive test for inline register constraints
;
; RUN: llc -march=mipsel < %s  | FileCheck -check-prefix=CHECK_LITTLE_32 %s
; RUN: llc -march=mips < %s  | FileCheck -check-prefix=CHECK_BIG_32 %s

%union.u_tag = type { i64 }
%struct.anon = type { i32, i32 }
@uval = common global %union.u_tag zeroinitializer, align 8

; X with -3
define i32 @constraint_X() nounwind {
entry:
;CHECK_LITTLE_32:   constraint_X:
;CHECK_LITTLE_32: #APP
;CHECK_LITTLE_32: addi ${{[0-9]+}},${{[0-9]+}},0xfffffffffffffffd
;CHECK_LITTLE_32: #NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:X}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; x with -3
define i32 @constraint_x() nounwind {
entry:
;CHECK_LITTLE_32:   constraint_x:
;CHECK_LITTLE_32: #APP
;CHECK_LITTLE_32: addi ${{[0-9]+}},${{[0-9]+}},0xfffd
;CHECK_LITTLE_32: #NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:x}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; d with -3
define i32 @constraint_d() nounwind {
entry:
;CHECK_LITTLE_32:   constraint_d:
;CHECK_LITTLE_32:   #APP
;CHECK_LITTLE_32:   addi ${{[0-9]+}},${{[0-9]+}},-3
;CHECK_LITTLE_32:   #NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:d}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; m with -3
define i32 @constraint_m() nounwind {
entry:
;CHECK_LITTLE_32:   constraint_m:
;CHECK_LITTLE_32:   #APP
;CHECK_LITTLE_32:   addi ${{[0-9]+}},${{[0-9]+}},-4
;CHECK_LITTLE_32:   #NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:m}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; z with -3
define i32 @constraint_z() nounwind {
entry:
;CHECK_LITTLE_32: constraint_z:
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    addi ${{[0-9]+}},${{[0-9]+}},-3
;CHECK_LITTLE_32:    #NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:z}", "=r,r,I"(i32 7, i32 -3) ;

; z with 0
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    addi ${{[0-9]+}},${{[0-9]+}},$0
;CHECK_LITTLE_32:    #NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:z}", "=r,r,I"(i32 7, i32 0) nounwind
  ret i32 0
}

; a long long in 32 bit mode (use to assert)
define i32 @constraint_longlong() nounwind {
entry:
;CHECK_LITTLE_32: constraint_longlong:
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    addi ${{[0-9]+}},${{[0-9]+}},3
;CHECK_LITTLE_32:    #NO_APP
  tail call i64 asm sideeffect "addi $0,$1,$2 \0A\09", "=r,r,X"(i64 1229801703532086340, i64 3) nounwind
  ret i32 0
}

; D, in little endian the source reg will be 4 bytes into the long long
define i32 @constraint_D() nounwind {
entry:
;CHECK_LITTLE_32: constraint_D:
;CHECK_LITTLE_32:    lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;CHECK_LITTLE_32:    lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;CHECK_LITTLE_32:    lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    or ${{[0-9]+}},$[[SECOND]],${{[0-9]+}}
;CHECK_LITTLE_32:    #NO_APP

; D, in big endian the source reg will also be 4 bytes into the long long
;CHECK_BIG_32:    constraint_D:
;CHECK_BIG_32:       lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;CHECK_BIG_32:       lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;CHECK_BIG_32:       lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;CHECK_BIG_32:       #APP
;CHECK_BIG_32:       or ${{[0-9]+}},$[[SECOND]],${{[0-9]+}}
;CHECK_BIG_32:       #NO_APP
  %bosco = load i64* getelementptr inbounds (%union.u_tag* @uval, i32 0, i32 0), align 8
  %trunc1 = trunc i64 %bosco to i32
  tail call i32 asm sideeffect "or $0,${1:D},$2", "=r,r,r"(i64 %bosco, i32 %trunc1) nounwind
  ret i32 0
}

; L, in little endian the source reg will be 0 bytes into the long long
define i32 @constraint_L() nounwind {
entry:
;CHECK_LITTLE_32: constraint_L:
;CHECK_LITTLE_32:    lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;CHECK_LITTLE_32:    lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;CHECK_LITTLE_32:    lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    or ${{[0-9]+}},$[[FIRST]],${{[0-9]+}}
;CHECK_LITTLE_32:    #NO_APP
; L, in big endian the source reg will be 4 bytes into the long long
;CHECK_BIG_32: constraint_L:
;CHECK_BIG_32:       lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;CHECK_BIG_32:       lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;CHECK_BIG_32:       lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;CHECK_BIG_32:       #APP
;CHECK_BIG_32:       or ${{[0-9]+}},$[[SECOND]],${{[0-9]+}}
;CHECK_BIG_32:       #NO_APP
  %bosco = load i64* getelementptr inbounds (%union.u_tag* @uval, i32 0, i32 0), align 8
  %trunc1 = trunc i64 %bosco to i32
  tail call i32 asm sideeffect "or $0,${1:L},$2", "=r,r,r"(i64 %bosco, i32 %trunc1) nounwind
  ret i32 0
}

; M, in little endian the source reg will be 4 bytes into the long long
define i32 @constraint_M() nounwind {
entry:
;CHECK_LITTLE_32: constraint_M:
;CHECK_LITTLE_32:    lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;CHECK_LITTLE_32:    lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;CHECK_LITTLE_32:    lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    or ${{[0-9]+}},$[[SECOND]],${{[0-9]+}}
;CHECK_LITTLE_32:    #NO_APP
; M, in big endian the source reg will be 0 bytes into the long long
;CHECK_BIG_32:    constraint_M:
;CHECK_BIG_32:       lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;CHECK_BIG_32:       lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;CHECK_BIG_32:       lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;CHECK_BIG_32:       #APP
;CHECK_BIG_32:       or ${{[0-9]+}},$[[FIRST]],${{[0-9]+}}
;CHECK_BIG_32:       #NO_APP
  %bosco = load i64* getelementptr inbounds (%union.u_tag* @uval, i32 0, i32 0), align 8
  %trunc1 = trunc i64 %bosco to i32
  tail call i32 asm sideeffect "or $0,${1:M},$2", "=r,r,r"(i64 %bosco, i32 %trunc1) nounwind
  ret i32 0
}
