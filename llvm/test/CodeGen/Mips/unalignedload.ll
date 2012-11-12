; RUN: llc  < %s -march=mipsel  | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips    | FileCheck %s -check-prefix=CHECK-EB
%struct.S2 = type { %struct.S1, %struct.S1 }
%struct.S1 = type { i8, i8 }
%struct.S4 = type { [7 x i8] }

@s2 = common global %struct.S2 zeroinitializer, align 1
@s4 = common global %struct.S4 zeroinitializer, align 1

define void @foo1() nounwind {
entry:
; CHECK-EL: lbu ${{[0-9]+}}, 2($[[R0:[0-9]+]])
; CHECK-EL: lbu ${{[0-9]+}}, 3($[[R0]])
; CHECK-EL: jalr
; CHECK-EL: lwl $[[R1:[0-9]+]], 3($[[R2:[0-9]+]])
; CHECK-EL: lwr $[[R1]], 0($[[R2]])

; CHECK-EB: lbu ${{[0-9]+}}, 3($[[R0:[0-9]+]])
; CHECK-EB: lbu ${{[0-9]+}}, 2($[[R0]])
; CHECK-EB: jalr
; CHECK-EB: lwl $[[R1:[0-9]+]], 0($[[R2:[0-9]+]])
; CHECK-BE: lwr $[[R1]], 3($[[R2]])

  tail call void @foo2(%struct.S1* byval getelementptr inbounds (%struct.S2* @s2, i32 0, i32 1)) nounwind
  tail call void @foo4(%struct.S4* byval @s4) nounwind
  ret void
}

declare void @foo2(%struct.S1* byval)

declare void @foo4(%struct.S4* byval)
