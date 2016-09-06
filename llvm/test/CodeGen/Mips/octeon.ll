; RUN: llc -O1 < %s -march=mips64 -mcpu=octeon | FileCheck %s -check-prefixes=ALL,OCTEON
; RUN: llc -O1 < %s -march=mips64 -mcpu=mips64 | FileCheck %s -check-prefixes=ALL,MIPS64

define i64 @addi64(i64 %a, i64 %b) nounwind {
entry:
; ALL-LABEL: addi64:
; OCTEON: jr      $ra
; OCTEON: baddu   $2, $4, $5
; MIPS64: daddu   $[[T0:[0-9]+]], $4, $5
; MIPS64: jr      $ra
; MIPS64: andi    $2, $[[T0]], 255
  %add = add i64 %a, %b
  %and = and i64 %add, 255
  ret i64 %and
}

define i64 @mul(i64 %a, i64 %b) nounwind {
entry:
; ALL-LABEL: mul:
; OCTEON: jr    $ra
; OCTEON: dmul  $2, $4, $5
; MIPS64: dmult $4, $5
; MIPS64: jr    $ra
; MIPS64: mflo  $2
  %res = mul i64 %a, %b
  ret i64 %res
}

define i64 @cmpeq(i64 %a, i64 %b) nounwind {
entry:
; ALL-LABEL: cmpeq:
; OCTEON: jr     $ra
; OCTEON: seq    $2, $4, $5
; MIPS64: xor    $[[T0:[0-9]+]], $4, $5
; MIPS64: sltiu  $[[T1:[0-9]+]], $[[T0]], 1
; MIPS64: dsll   $[[T2:[0-9]+]], $[[T1]], 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $[[T2]], 32
  %res = icmp eq i64 %a, %b
  %res2 = zext i1 %res to i64
  ret i64 %res2
}

define i64 @cmpeqi(i64 %a) nounwind {
entry:
; ALL-LABEL: cmpeqi:
; OCTEON: jr     $ra
; OCTEON: seqi   $2, $4, 42
; MIPS64: daddiu $[[T0:[0-9]+]], $zero, 42
; MIPS64: xor    $[[T1:[0-9]+]], $4, $[[T0]]
; MIPS64: sltiu  $[[T2:[0-9]+]], $[[T1]], 1
; MIPS64: dsll   $[[T3:[0-9]+]], $[[T2]], 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $[[T3]], 32
  %res = icmp eq i64 %a, 42
  %res2 = zext i1 %res to i64
  ret i64 %res2
}

define i64 @cmpne(i64 %a, i64 %b) nounwind {
entry:
; ALL-LABEL: cmpne:
; OCTEON: jr     $ra
; OCTEON: sne    $2, $4, $5
; MIPS64: xor    $[[T0:[0-9]+]], $4, $5
; MIPS64: sltu   $[[T1:[0-9]+]], $zero, $[[T0]]
; MIPS64: dsll   $[[T2:[0-9]+]], $[[T1]], 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $[[T2]], 32
  %res = icmp ne i64 %a, %b
  %res2 = zext i1 %res to i64
  ret i64 %res2
}

define i64 @cmpnei(i64 %a) nounwind {
entry:
; ALL-LABEL: cmpnei:
; OCTEON: jr     $ra
; OCTEON: snei   $2, $4, 42
; MIPS64: daddiu $[[T0:[0-9]+]], $zero, 42
; MIPS64: xor    $[[T1:[0-9]+]], $4, $[[T0]]
; MIPS64: sltu   $[[T2:[0-9]+]], $zero, $[[T1]]
; MIPS64: dsll   $[[T3:[0-9]+]], $[[T2]], 32
; MIPS64: jr     $ra
; MIPS64: dsrl   $2, $[[T3]], 32
  %res = icmp ne i64 %a, 42
  %res2 = zext i1 %res to i64
  ret i64 %res2
}

define i64 @bbit0(i64 %a) nounwind {
entry:
; ALL-LABEL: bbit0:
; OCTEON: bbit0   $4, 3, [[BB0:(\$|\.L)BB[0-9_]+]]
; MIPS64: andi  $[[T0:[0-9]+]], $4, 8
; MIPS64: bnez  $[[T0]], [[BB0:(\$|\.L)BB[0-9_]+]]
  %bit = and i64 %a, 8
  %res = icmp eq i64 %bit, 0
  br i1 %res, label %endif, label %if
if:
  ret i64 48

endif:
  ret i64 12
}

define i64 @bbit032(i64 %a) nounwind {
entry:
; ALL-LABEL: bbit032:
; OCTEON: bbit032 $4, 3, [[BB0:(\$|\.L)BB[0-9_]+]]
; MIPS64: daddiu  $[[T0:[0-9]+]], $zero, 1
; MIPS64: dsll    $[[T1:[0-9]+]], $[[T0]], 35
; MIPS64: and     $[[T2:[0-9]+]], $4, $[[T1]]
; MIPS64: bnez    $[[T2]], [[BB0:(\$|\.L)BB[0-9_]+]]
  %bit = and i64 %a, 34359738368
  %res = icmp eq i64 %bit, 0
  br i1 %res, label %endif, label %if
if:
  ret i64 48

endif:
  ret i64 12
}

define i64 @bbit1(i64 %a) nounwind {
entry:
; ALL-LABEL: bbit1:
; OCTEON: bbit1 $4, 3, [[BB0:(\$|\.L)BB[0-9_]+]]
; MIPS64: andi  $[[T0:[0-9]+]], $4, 8
; MIPS64: beqz  $[[T0]], [[BB0:(\$|\.L)BB[0-9_]+]]
  %bit = and i64 %a, 8
  %res = icmp ne i64 %bit, 0
  br i1 %res, label %endif, label %if
if:
  ret i64 48

endif:
  ret i64 12
}

define i64 @bbit132(i64 %a) nounwind {
entry:
; ALL-LABEL: bbit132:
; OCTEON: bbit132 $4, 3, [[BB0:(\$|\.L)BB[0-9_]+]]
; MIPS64: daddiu  $[[T0:[0-9]+]], $zero, 1
; MIPS64: dsll    $[[T1:[0-9]+]], $[[T0]], 35
; MIPS64: and     $[[T2:[0-9]+]], $4, $[[T1]]
; MIPS64: beqz    $[[T2]], [[BB0:(\$|\.L)BB[0-9_]+]]
  %bit = and i64 %a, 34359738368
  %res = icmp ne i64 %bit, 0
  br i1 %res, label %endif, label %if
if:
  ret i64 48

endif:
  ret i64 12
}
