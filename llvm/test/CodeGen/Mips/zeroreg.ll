; RUN: llc < %s -march=mipsel -mcpu=mips32   -relocation-model=pic | FileCheck %s -check-prefixes=ALL,32-CMOV
; RUN: llc < %s -march=mipsel -mcpu=mips32r2 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,32-CMOV
; RUN: llc < %s -march=mipsel -mcpu=mips32r6 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,32R6
; RUN: llc < %s -march=mipsel -mcpu=mips4    -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,64-CMOV
; RUN: llc < %s -march=mipsel -mcpu=mips64   -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,64-CMOV
; RUN: llc < %s -march=mipsel -mcpu=mips64r2 -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,64-CMOV
; RUN: llc < %s -march=mipsel -mcpu=mips64r6 -target-abi n64 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,64R6

@g1 = external global i32

define i32 @sel_icmp_nez_i32_z0(i32 signext %s) nounwind readonly {
entry:
; ALL-LABEL: sel_icmp_nez_i32_z0:

; 32-CMOV:       lw $2, 0(${{[0-9]+}})
; 32-CMOV:       movn $2, $zero, $4

; 32R6:          lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R6:          seleqz $2, $[[R0]], $4

; 64-CMOV:       lw $2, 0(${{[0-9]+}})
; 64-CMOV:       movn $2, $zero, $4

; 64R6:          lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 64R6:          seleqz $2, $[[R0]], $4

  %tobool = icmp ne i32 %s, 0
  %0 = load i32, ptr @g1, align 4
  %cond = select i1 %tobool, i32 0, i32 %0
  ret i32 %cond
}

define i32 @sel_icmp_nez_i32_z1(i32 signext %s) nounwind readonly {
entry:
; ALL-LABEL: sel_icmp_nez_i32_z1:

; 32-CMOV:       lw $2, 0(${{[0-9]+}})
; 32-CMOV:       movz $2, $zero, $4

; 32R6:          lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R6:          selnez $2, $[[R0]], $4

; 64-CMOV:       lw $2, 0(${{[0-9]+}})
; 64-CMOV:       movz $2, $zero, $4

; 64R6:          lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 64R6:          selnez $2, $[[R0]], $4

  %tobool = icmp ne i32 %s, 0
  %0 = load i32, ptr @g1, align 4
  %cond = select i1 %tobool, i32 %0, i32 0
  ret i32 %cond
}

@g2 = external global i64

define i64 @sel_icmp_nez_i64_z0(i64 %s) nounwind readonly {
entry:
; ALL-LABEL: sel_icmp_nez_i64_z0:

; 32-CMOV-DAG:   lw $[[R0:2]], 0(${{[0-9]+}})
; 32-CMOV-DAG:   lw $[[R1:3]], 4(${{[0-9]+}})
; 32-CMOV-DAG:   movn $[[R0]], $zero, $4
; 32-CMOV-DAG:   movn $[[R1]], $zero, $4

; 32R6-DAG:      lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R6-DAG:      lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R6-DAG:      or $[[CC:[0-9]+]], $4, $5
; 32R6-DAG:      seleqz $2, $[[R0]], $[[CC]]
; 32R6-DAG:      seleqz $3, $[[R1]], $[[CC]]

; 64-CMOV:       ld $2, 0(${{[0-9]+}})
; 64-CMOV:       movn $2, $zero, $4

; 64R6:          ld $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 64R6:          seleqz $2, $[[R0]], $4

  %tobool = icmp ne i64 %s, 0
  %0 = load i64, ptr @g2, align 4
  %cond = select i1 %tobool, i64 0, i64 %0
  ret i64 %cond
}

define i64 @sel_icmp_nez_i64_z1(i64 %s) nounwind readonly {
entry:
; ALL-LABEL: sel_icmp_nez_i64_z1:

; 32-CMOV-DAG:   lw $[[R0:2]], 0(${{[0-9]+}})
; 32-CMOV-DAG:   lw $[[R1:3]], 4(${{[0-9]+}})
; 32-CMOV-DAG:   movz $[[R0]], $zero, $4
; 32-CMOV-DAG:   movz $[[R1]], $zero, $4

; 32R6-DAG:      lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R6-DAG:      lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R6-DAG:      or $[[CC:[0-9]+]], $4, $5
; 32R6-DAG:      selnez $2, $[[R0]], $[[CC]]
; 32R6-DAG:      selnez $3, $[[R1]], $[[CC]]

; 64-CMOV:       ld $2, 0(${{[0-9]+}})
; 64-CMOV:       movz $2, $zero, $4

; 64R6:          ld $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 64R6:          selnez $2, $[[R0]], $4

  %tobool = icmp ne i64 %s, 0
  %0 = load i64, ptr @g2, align 4
  %cond = select i1 %tobool, i64 %0, i64 0
  ret i64 %cond
}
