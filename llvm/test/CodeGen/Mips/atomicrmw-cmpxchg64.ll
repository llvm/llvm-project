; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=MIPS32,MIPS32R2
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=MIPS32,MIPS32R6
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIPS64
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIPS64

define i64 @uinc_wrap_i64(ptr %ptr, i64 %value) {
; MIPS64-LABEL: uinc_wrap_i64:
; MIPS64: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: {{^[ \t]+}}lld $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; MIPS64-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; MIPS64-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; MIPS64: {{^[ \t]+}}scd $[[SC:[0-9]+]], 0($[[PTR]])
; MIPS64-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; MIPS64: [[CMPDONE]]:
; MIPS64: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS32-LABEL: uinc_wrap_i64:
; MIPS32: {{^[ \t]+}}jal __atomic_load_8
; MIPS32: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS32: {{^[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED:]]($sp)
; MIPS32-NEXT: {{[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32: {{^[ \t]+}}jal __atomic_compare_exchange_8
; MIPS32R2: {{^[ \t]+}}move $[[SUCCESS:[0-9]+]], $2
; MIPS32: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32R2: {{^[ \t]+}}beqz $[[SUCCESS]], [[RETRY]]
; MIPS32R2-NEXT: {{[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}beqzc $2, [[RETRY]]
  %old = atomicrmw uinc_wrap ptr %ptr, i64 %value seq_cst
  ret i64 %old
}

define i64 @udec_wrap_i64(ptr %ptr, i64 %value) {
; MIPS64-LABEL: udec_wrap_i64:
; MIPS64: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: {{^[ \t]+}}lld $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; MIPS64-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; MIPS64-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; MIPS64: {{^[ \t]+}}scd $[[SC:[0-9]+]], 0($[[PTR]])
; MIPS64-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; MIPS64: [[CMPDONE]]:
; MIPS64: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS32-LABEL: udec_wrap_i64:
; MIPS32: {{^[ \t]+}}jal __atomic_load_8
; MIPS32: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS32: {{^[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED:]]($sp)
; MIPS32-NEXT: {{[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32: {{^[ \t]+}}jal __atomic_compare_exchange_8
; MIPS32R2: {{^[ \t]+}}move $[[SUCCESS:[0-9]+]], $2
; MIPS32: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32R2: {{^[ \t]+}}beqz $[[SUCCESS]], [[RETRY]]
; MIPS32R2-NEXT: {{[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}beqzc $2, [[RETRY]]
  %old = atomicrmw udec_wrap ptr %ptr, i64 %value seq_cst
  ret i64 %old
}

define i64 @usub_cond_i64(ptr %ptr, i64 %value) {
; MIPS64-LABEL: usub_cond_i64:
; MIPS64: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: {{^[ \t]+}}lld $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; MIPS64-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; MIPS64-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; MIPS64: {{^[ \t]+}}scd $[[SC:[0-9]+]], 0($[[PTR]])
; MIPS64-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; MIPS64: [[CMPDONE]]:
; MIPS64: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS32-LABEL: usub_cond_i64:
; MIPS32: {{^[ \t]+}}jal __atomic_load_8
; MIPS32: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS32: {{^[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED:]]($sp)
; MIPS32-NEXT: {{[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32: {{^[ \t]+}}jal __atomic_compare_exchange_8
; MIPS32: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32R2: {{^[ \t]+}}beqz $2, [[RETRY]]
; MIPS32R2-NEXT: {{[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}beqzc $2, [[RETRY]]
  %old = atomicrmw usub_cond ptr %ptr, i64 %value seq_cst
  ret i64 %old
}

define i64 @usub_sat_i64(ptr %ptr, i64 %value) {
; MIPS64-LABEL: usub_sat_i64:
; MIPS64: [[RMW:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: [[LLSC:(\$BB|\.LBB)[0-9]+_[0-9]+]]:
; MIPS64: {{^[ \t]+}}lld $[[OLD:[0-9]+]], 0($[[PTR:[0-9]+]])
; MIPS64-NEXT: {{[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP:[0-9]+]], [[CMPDONE:(\$BB|\.LBB)[0-9]+_[0-9]+]]
; MIPS64-NOT: {{^[ \t]+s[bhwd]([lr]|c[0-3])?[ \t]}}
; MIPS64: {{^[ \t]+}}scd $[[SC:[0-9]+]], 0($[[PTR]])
; MIPS64-NEXT: {{[ \t]+}}beqz{{c?}} $[[SC]], [[LLSC]]
; MIPS64: [[CMPDONE]]:
; MIPS64: {{^[ \t]+}}bne{{c?}} $[[OLD]], $[[CMP]], [[RMW]]
; MIPS32-LABEL: usub_sat_i64:
; MIPS32: {{^[ \t]+}}jal __atomic_load_8
; MIPS32: [[RETRY:\$BB[0-9]+_[0-9]+]]:
; MIPS32: {{^[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED:]]($sp)
; MIPS32-NEXT: {{[ \t]+}}sw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32: {{^[ \t]+}}jal __atomic_compare_exchange_8
; MIPS32R2: {{^[ \t]+}}move $[[SUCCESS:[0-9]+]], $2
; MIPS32: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED + 4]]($sp)
; MIPS32R2: {{^[ \t]+}}beqz $[[SUCCESS]], [[RETRY]]
; MIPS32R2-NEXT: {{[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}lw ${{[0-9]+}}, [[#EXPECTED]]($sp)
; MIPS32R6: {{^[ \t]+}}beqzc $2, [[RETRY]]
  %old = atomicrmw usub_sat ptr %ptr, i64 %value seq_cst
  ret i64 %old
}
