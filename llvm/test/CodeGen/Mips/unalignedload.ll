; RUN: llc  < %s -march=mipsel -mcpu=mips32   -relocation-model=pic | FileCheck %s -check-prefixes=ALL,MIPS32-EL
; RUN: llc  < %s -march=mips   -mcpu=mips32   -relocation-model=pic | FileCheck %s -check-prefixes=ALL,MIPS32-EB
; RUN: llc  < %s -march=mipsel -mcpu=mips32r2 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,MIPS32-EL
; RUN: llc  < %s -march=mips   -mcpu=mips32r2 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,MIPS32-EB
; RUN: llc  < %s -march=mipsel -mcpu=mips32r6 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,MIPS32R6-EL
; RUN: llc  < %s -march=mips   -mcpu=mips32r6 -relocation-model=pic | FileCheck %s -check-prefixes=ALL,MIPS32R6-EB
%struct.S2 = type { %struct.S1, %struct.S1 }
%struct.S1 = type { i8, i8 }
%struct.S4 = type { [7 x i8] }

@s2 = common global %struct.S2 zeroinitializer, align 1
@s4 = common global %struct.S4 zeroinitializer, align 1

define void @bar1() nounwind {
entry:
; ALL-LABEL: bar1:

; ALL-DAG:       lw $[[R0:[0-9]+]], %got(s2)(

; MIPS32-EL-DAG: lbu $[[PART1:[0-9]+]], 2($[[R0]])
; MIPS32-EL-DAG: lbu $[[PART2:[0-9]+]], 3($[[R0]])
; MIPS32-EL-DAG: sll $[[T0:[0-9]+]], $[[PART2]], 8
; MIPS32-EL-DAG: or  $4, $[[T0]], $[[PART1]]

; MIPS32-EB-DAG: lbu $[[PART1:[0-9]+]], 2($[[R0]])
; MIPS32-EB-DAG: lbu $[[PART2:[0-9]+]], 3($[[R0]])
; MIPS32-EB-DAG: sll $[[T0:[0-9]+]], $[[PART1]], 8
; MIPS32-EB-DAG: or  $[[T1:[0-9]+]], $[[T0]], $[[PART2]]
; MIPS32-EB-DAG: sll $4, $[[T1]], 16

; MIPS32R6-DAG:  lhu $[[PART1:[0-9]+]], 2($[[R0]])

  tail call void @foo2(%struct.S1* byval getelementptr inbounds (%struct.S2, %struct.S2* @s2, i32 0, i32 1)) nounwind
  ret void
}

define void @bar2() nounwind {
entry:
; ALL-LABEL: bar2:

; ALL-DAG:       lw $[[R2:[0-9]+]], %got(s4)(

; MIPS32-EL-DAG: lwl $[[R1:4]], 3($[[R2]])
; MIPS32-EL-DAG: lwr $[[R1]], 0($[[R2]])
; MIPS32-EL-DAG: lbu $[[T0:[0-9]+]], 4($[[R2]])
; MIPS32-EL-DAG: lbu $[[T1:[0-9]+]], 5($[[R2]])
; MIPS32-EL-DAG: lbu $[[T2:[0-9]+]], 6($[[R2]])
; MIPS32-EL-DAG: sll $[[T3:[0-9]+]], $[[T1]], 8
; MIPS32-EL-DAG: or  $[[T4:[0-9]+]], $[[T3]], $[[T0]]
; MIPS32-EL-DAG: sll $[[T5:[0-9]+]], $[[T2]], 16
; MIPS32-EL-DAG: or  $5, $[[T4]], $[[T5]]

; MIPS32-EB-DAG: lwl $[[R1:4]], 0($[[R2]])
; MIPS32-EB-DAG: lwr $[[R1]], 3($[[R2]])
; MIPS32-EB-DAG: lbu $[[T0:[0-9]+]], 4($[[R2]])
; MIPS32-EB-DAG: lbu $[[T1:[0-9]+]], 5($[[R2]])
; MIPS32-EB-DAG: lbu $[[T2:[0-9]+]], 6($[[R2]])
; MIPS32-EB-DAG: sll $[[T3:[0-9]+]], $[[T0]], 8
; MIPS32-EB-DAG: or  $[[T4:[0-9]+]], $[[T3]], $[[T1]]
; MIPS32-EB-DAG: sll $[[T5:[0-9]+]], $[[T4]], 16
; MIPS32-EB-DAG: sll $[[T6:[0-9]+]], $[[T2]], 8
; MIPS32-EB-DAG: or  $5, $[[T5]], $[[T6]]

; FIXME: We should be able to do better than this using lhu
; MIPS32R6-EL-DAG: lw $4, 0($[[R2]])
; MIPS32R6-EL-DAG: lhu $[[T0:[0-9]+]], 4($[[R2]])
; MIPS32R6-EL-DAG: lbu $[[T1:[0-9]+]], 6($[[R2]])
; MIPS32R6-EL-DAG: sll $[[T2:[0-9]+]], $[[T1]], 16
; MIPS32R6-EL-DAG: or  $5, $[[T0]], $[[T2]]

; FIXME: We should be able to do better than this using lhu
; MIPS32R6-EB-DAG: lw $4, 0($[[R2]])
; MIPS32R6-EB-DAG: lhu $[[T0:[0-9]+]], 4($[[R2]])
; MIPS32R6-EB-DAG: lbu $[[T1:[0-9]+]], 6($[[R2]])
; MIPS32R6-EB-DAG: sll $[[T2:[0-9]+]], $[[T0]], 16
; MIPS32R6-EB-DAG: sll $[[T3:[0-9]+]], $[[T1]], 8
; MIPS32R6-EB-DAG: or  $5, $[[T2]], $[[T3]]

  tail call void @foo4(%struct.S4* byval @s4) nounwind
  ret void
}

declare void @foo2(%struct.S1* byval)

declare void @foo4(%struct.S4* byval)
