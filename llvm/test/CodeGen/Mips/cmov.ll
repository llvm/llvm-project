; RUN: llc -march=mips     -mcpu=mips32                 < %s | FileCheck %s -check-prefix=ALL -check-prefix=32-CMOV
; RUN: llc -march=mips     -mcpu=mips32 -regalloc=basic < %s | FileCheck %s -check-prefix=ALL -check-prefix=32-CMOV
; RUN: llc -march=mips     -mcpu=mips32r2               < %s | FileCheck %s -check-prefix=ALL -check-prefix=32-CMOV
; RUN: llc -march=mips     -mcpu=mips32r6               < %s | FileCheck %s -check-prefix=ALL -check-prefix=32-CMP
; RUN: llc -march=mips64el -mcpu=mips4                  < %s | FileCheck %s -check-prefix=ALL -check-prefix=64-CMOV
; RUN: llc -march=mips64el -mcpu=mips64                 < %s | FileCheck %s -check-prefix=ALL -check-prefix=64-CMOV
; RUN: llc -march=mips64el -mcpu=mips64r6               < %s | FileCheck %s -check-prefix=ALL -check-prefix=64-CMP

@i1 = global [3 x i32] [i32 1, i32 2, i32 3], align 4
@i3 = common global i32* null, align 4

; ALL-LABEL: cmov1:

; 32-CMOV-DAG:  lw $[[R0:[0-9]+]], %got(i3)
; 32-CMOV-DAG:  addiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got(i1)
; 32-CMOV-DAG:  movn $[[R0]], $[[R1]], $4
; 32-CMOV-DAG:  lw $2, 0($[[R0]])

; 32-CMP-DAG:   lw $[[R0:[0-9]+]], %got(i3)
; 32-CMP-DAG:   addiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got(i1)
; 32-CMP-DAG:   selnez $[[T0:[0-9]+]], $[[R1]], $4
; 32-CMP-DAG:   seleqz $[[T1:[0-9]+]], $[[R0]], $4
; 32-CMP-DAG:   or $[[T2:[0-9]+]], $[[T0]], $[[T1]]
; 32-CMP-DAG:   lw $2, 0($[[T2]])

; 64-CMOV-DAG:  ldr $[[R0:[0-9]+]]
; 64-CMOV-DAG:  ld $[[R1:[0-9]+]], %got_disp(i1)
; 64-CMOV-DAG:  movn $[[R0]], $[[R1]], $4

; 64-CMP-DAG:   ld $[[R0:[0-9]+]], %got_disp(i3)(
; 64-CMP-DAG:   daddiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got_disp(i1)
; FIXME: This sll works around an implementation detail in the code generator
;        (setcc's result is i32 so bits 32-63 are undefined). It's not really
;        needed.
; 64-CMP-DAG:   sll $[[CC:[0-9]+]], $4, 0
; 64-CMP-DAG:   selnez $[[T0:[0-9]+]], $[[R1]], $[[CC]]
; 64-CMP-DAG:   seleqz $[[T1:[0-9]+]], $[[R0]], $[[CC]]
; 64-CMP-DAG:   or $[[T2:[0-9]+]], $[[T0]], $[[T1]]
; 64-CMP-DAG:   ld $2, 0($[[T2]])

define i32* @cmov1(i32 signext %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32*, i32** @i3, align 4
  %cond = select i1 %tobool, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @i1, i32 0, i32 0), i32* %tmp1
  ret i32* %cond
}

@c = global i32 1, align 4
@d = global i32 0, align 4

; ALL-LABEL: cmov2:

; 32-CMOV-DAG:  addiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got(d)
; 32-CMOV-DAG:  addiu $[[R0:[0-9]+]], ${{[0-9]+}}, %got(c)
; 32-CMOV-DAG:  movn  $[[R1]], $[[R0]], $4
; 32-CMOV-DAG:  lw $2, 0($[[R0]])

; 32-CMP-DAG:   addiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got(d)
; 32-CMP-DAG:   addiu $[[R0:[0-9]+]], ${{[0-9]+}}, %got(c)
; 32-CMP-DAG:   selnez $[[T0:[0-9]+]], $[[R0]], $4
; 32-CMP-DAG:   seleqz $[[T1:[0-9]+]], $[[R1]], $4
; 32-CMP-DAG:   or $[[T2:[0-9]+]], $[[T0]], $[[T1]]
; 32-CMP-DAG:   lw $2, 0($[[T2]])

; 64-CMOV:      daddiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got_disp(d)
; 64-CMOV:      daddiu $[[R0:[0-9]+]], ${{[0-9]+}}, %got_disp(c)
; 64-CMOV:      movn  $[[R1]], $[[R0]], $4

; 64-CMP-DAG:   daddiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got_disp(d)
; 64-CMP-DAG:   daddiu $[[R0:[0-9]+]], ${{[0-9]+}}, %got_disp(c)
; FIXME: This sll works around an implementation detail in the code generator
;        (setcc's result is i32 so bits 32-63 are undefined). It's not really
;        needed.
; 64-CMP-DAG:   sll $[[CC:[0-9]+]], $4, 0
; 64-CMP-DAG:   selnez $[[T0:[0-9]+]], $[[R0]], $[[CC]]
; 64-CMP-DAG:   seleqz $[[T1:[0-9]+]], $[[R1]], $[[CC]]
; 64-CMP-DAG:   or $[[T2:[0-9]+]], $[[T0]], $[[T1]]
; 64-CMP-DAG:   lw $2, 0($[[T2]])

define i32 @cmov2(i32 signext %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32, i32* @c, align 4
  %tmp2 = load i32, i32* @d, align 4
  %cond = select i1 %tobool, i32 %tmp1, i32 %tmp2
  ret i32 %cond
}

; ALL-LABEL: cmov3:

; We won't check the result register since we can't know if the move is first
; or last. We do know it will be either one of two registers so we can at least
; check that.

; 32-CMOV:      xori $[[R0:[0-9]+]], $4, 234
; 32-CMOV:      movz ${{[26]}}, $5, $[[R0]]

; 32-CMP-DAG:   xori $[[CC:[0-9]+]], $4, 234
; 32-CMP-DAG:   seleqz $[[T0:[0-9]+]], $5, $[[CC]]
; 32-CMP-DAG:   selnez $[[T1:[0-9]+]], $6, $[[CC]]
; 32-CMP-DAG:   or $2, $[[T0]], $[[T1]]

; 64-CMOV:      xori $[[R0:[0-9]+]], $4, 234
; 64-CMOV:      movz ${{[26]}}, $5, $[[R0]]

; 64-CMP-DAG:   xori $[[CC:[0-9]+]], $4, 234
; 64-CMP-DAG:   seleqz $[[T0:[0-9]+]], $5, $[[CC]]
; 64-CMP-DAG:   selnez $[[T1:[0-9]+]], $6, $[[CC]]
; 64-CMP-DAG:   or $2, $[[T0]], $[[T1]]

define i32 @cmov3(i32 signext %a, i32 signext %b, i32 signext %c) nounwind readnone {
entry:
  %cmp = icmp eq i32 %a, 234
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; ALL-LABEL: cmov3_ne:

; We won't check the result register since we can't know if the move is first
; or last. We do know it will be either one of two registers so we can at least
; check that.

; FIXME: Use xori instead of addiu+xor.
; 32-CMOV:      addiu $[[R0:[0-9]+]], $zero, 234
; 32-CMOV:      xor $[[R1:[0-9]+]], $4, $[[R0]]
; 32-CMOV:      movn ${{[26]}}, $5, $[[R1]]

; 32-CMP-DAG:   xori $[[CC:[0-9]+]], $4, 234
; 32-CMP-DAG:   selnez $[[T0:[0-9]+]], $5, $[[CC]]
; 32-CMP-DAG:   seleqz $[[T1:[0-9]+]], $6, $[[CC]]
; 32-CMP-DAG:   or $2, $[[T0]], $[[T1]]

; FIXME: Use xori instead of addiu+xor.
; 64-CMOV:      addiu $[[R0:[0-9]+]], $zero, 234
; 64-CMOV:      xor $[[R1:[0-9]+]], $4, $[[R0]]
; 64-CMOV:      movn ${{[26]}}, $5, $[[R1]]

; 64-CMP-DAG:   xori $[[CC:[0-9]+]], $4, 234
; 64-CMP-DAG:   selnez $[[T0:[0-9]+]], $5, $[[CC]]
; 64-CMP-DAG:   seleqz $[[T1:[0-9]+]], $6, $[[CC]]
; 64-CMP-DAG:   or $2, $[[T0]], $[[T1]]

define i32 @cmov3_ne(i32 signext %a, i32 signext %b, i32 signext %c) nounwind readnone {
entry:
  %cmp = icmp ne i32 %a, 234
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; ALL-LABEL: cmov4:

; We won't check the result register since we can't know if the move is first
; or last. We do know it will be one of two registers so we can at least check
; that.

; 32-CMOV-DAG: xori $[[R0:[0-9]+]], $4, 234
; 32-CMOV-DAG: lw $[[R1:2]], 16($sp)
; 32-CMOV-DAG: lw $[[R2:3]], 20($sp)
; 32-CMOV-DAG: movz $[[R1]], $6, $[[R0]]
; 32-CMOV-DAG: movz $[[R2]], $7, $[[R0]]

; 32-CMP-DAG:  xori $[[R0:[0-9]+]], $4, 234
; 32-CMP-DAG:  lw $[[R1:[0-9]+]], 16($sp)
; 32-CMP-DAG:  lw $[[R2:[0-9]+]], 20($sp)
; 32-CMP-DAG:  seleqz $[[T0:[0-9]+]], $6, $[[R0]]
; 32-CMP-DAG:  seleqz $[[T1:[0-9]+]], $7, $[[R0]]
; 32-CMP-DAG:  selnez $[[T2:[0-9]+]], $[[R1]], $[[R0]]
; 32-CMP-DAG:  selnez $[[T3:[0-9]+]], $[[R2]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T2]]
; 32-CMP-DAG:  or $3, $[[T1]], $[[T3]]

; 64-CMOV: xori $[[R0:[0-9]+]], $4, 234
; 64-CMOV: movz ${{[26]}}, $5, $[[R0]]

; 64-CMP-DAG:  xori $[[R0:[0-9]+]], $4, 234
; 64-CMP-DAG:  seleqz $[[T0:[0-9]+]], $5, $[[R0]]
; 64-CMP-DAG:  selnez $[[T1:[0-9]+]], $6, $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i64 @cmov4(i32 signext %a, i64 %b, i64 %c) nounwind readnone {
entry:
  %cmp = icmp eq i32 %a, 234
  %cond = select i1 %cmp, i64 %b, i64 %c
  ret i64 %cond
}

; ALL-LABEL: cmov4_ne:

; We won't check the result register since we can't know if the move is first
; or last. We do know it will be one of two registers so we can at least check
; that.

; FIXME: Use xori instead of addiu+xor.
; 32-CMOV-DAG: addiu $[[R0:[0-9]+]], $zero, 234
; 32-CMOV-DAG: xor $[[R1:[0-9]+]], $4, $[[R0]]
; 32-CMOV-DAG: lw $[[R2:2]], 16($sp)
; 32-CMOV-DAG: lw $[[R3:3]], 20($sp)
; 32-CMOV-DAG: movn $[[R2]], $6, $[[R1]]
; 32-CMOV-DAG: movn $[[R3]], $7, $[[R1]]

; 32-CMP-DAG:  xori $[[R0:[0-9]+]], $4, 234
; 32-CMP-DAG:  lw $[[R1:[0-9]+]], 16($sp)
; 32-CMP-DAG:  lw $[[R2:[0-9]+]], 20($sp)
; 32-CMP-DAG:  selnez $[[T0:[0-9]+]], $6, $[[R0]]
; 32-CMP-DAG:  selnez $[[T1:[0-9]+]], $7, $[[R0]]
; 32-CMP-DAG:  seleqz $[[T2:[0-9]+]], $[[R1]], $[[R0]]
; 32-CMP-DAG:  seleqz $[[T3:[0-9]+]], $[[R2]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T2]]
; 32-CMP-DAG:  or $3, $[[T1]], $[[T3]]

; FIXME: Use xori instead of addiu+xor.
; 64-CMOV: addiu $[[R0:[0-9]+]], $zero, 234
; 64-CMOV: xor $[[R1:[0-9]+]], $4, $[[R0]]
; 64-CMOV: movn ${{[26]}}, $5, $[[R1]]

; 64-CMP-DAG:  xori $[[R0:[0-9]+]], $4, 234
; 64-CMP-DAG:  selnez $[[T0:[0-9]+]], $5, $[[R0]]
; 64-CMP-DAG:  seleqz $[[T1:[0-9]+]], $6, $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i64 @cmov4_ne(i32 signext %a, i64 %b, i64 %c) nounwind readnone {
entry:
  %cmp = icmp ne i32 %a, 234
  %cond = select i1 %cmp, i64 %b, i64 %c
  ret i64 %cond
}

; slti and conditional move.
;
; Check that, pattern
;  (select (setgt a, N), t, f)
; turns into
;  (movz t, (setlt a, N + 1), f)
; if N + 1 fits in 16-bit.

; ALL-LABEL: slti0:

; 32-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: slti $[[R0:[0-9]+]], $4, 32767
; 32-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  slti $[[R0:[0-9]+]], $4, 32767
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 32-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 32-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: slti $[[R0:[0-9]+]], $4, 32767
; 64-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 64-CMP-DAG:  slti $[[R0:[0-9]+]], $4, 32767
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 64-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 64-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @slti0(i32 signext %a) {
entry:
  %cmp = icmp sgt i32 %a, 32766
  %cond = select i1 %cmp, i32 3, i32 5
  ret i32 %cond
}

; ALL-LABEL: slti1:

; 32-CMOV-DAG: addiu $[[I7:[0-9]+]], $zero, 7
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: addiu $[[R1:[0-9]+]], $zero, 32767
; 32-CMOV-DAG: slt $[[R0:[0-9]+]], $[[R1]], $4
; 32-CMOV-DAG: movn $[[I5]], $[[I7]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I7:[0-9]+]], $zero, 7
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  addiu $[[I32767:[0-9]+]], $zero, 32767
; 32-CMP-DAG:  slt $[[R0:[0-9]+]], $[[I32767]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 32-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I7]], $[[R0]]
; 32-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I7:[0-9]+]], $zero, 7
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: addiu $[[R1:[0-9]+]], $zero, 32767
; 64-CMOV-DAG: slt $[[R0:[0-9]+]], $[[R1]], $4
; 64-CMOV-DAG: movn $[[I5]], $[[I7]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I7:[0-9]+]], $zero, 7
; 64-CMP-DAG:  addiu $[[I5:2]], $zero, 5
; 64-CMP-DAG:  addiu $[[R1:[0-9]+]], $zero, 32767
; 64-CMP-DAG:  slt $[[R0:[0-9]+]], $[[R1]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 64-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I7]], $[[R0]]
; 64-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @slti1(i32 signext %a) {
entry:
  %cmp = icmp sgt i32 %a, 32767
  %cond = select i1 %cmp, i32 7, i32 5
  ret i32 %cond
}

; ALL-LABEL: slti2:

; 32-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: slti $[[R0:[0-9]+]], $4, -32768
; 32-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  slti $[[R0:[0-9]+]], $4, -32768
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 32-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 32-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: slti $[[R0:[0-9]+]], $4, -32768
; 64-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 64-CMP-DAG:  slti $[[R0:[0-9]+]], $4, -32768
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 64-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 64-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @slti2(i32 signext %a) {
entry:
  %cmp = icmp sgt i32 %a, -32769
  %cond = select i1 %cmp, i32 3, i32 5
  ret i32 %cond
}

; ALL-LABEL: slti3:

; 32-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: lui $[[R1:[0-9]+]], 65535
; 32-CMOV-DAG: ori $[[R1]], $[[R1]], 32766
; 32-CMOV-DAG: slt $[[R0:[0-9]+]], $[[R1]], $4
; 32-CMOV-DAG: movn $[[I5]], $[[I3]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  lui $[[IMM:[0-9]+]], 65535
; 32-CMP-DAG:  ori $[[IMM]], $[[IMM]], 32766
; 32-CMP-DAG:  slt $[[R0:[0-9]+]], $[[I32767]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 32-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 32-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: lui $[[R1:[0-9]+]], 65535
; 64-CMOV-DAG: ori $[[R1]], $[[R1]], 32766
; 64-CMOV-DAG: slt $[[R0:[0-9]+]], $[[R1]], $4
; 64-CMOV-DAG: movn $[[I5]], $[[I3]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMP-DAG:  addiu $[[I5:2]], $zero, 5
; 64-CMP-DAG:  lui $[[IMM:[0-9]+]], 65535
; 64-CMP-DAG:  ori $[[IMM]], $[[IMM]], 32766
; 64-CMP-DAG:  slt $[[R0:[0-9]+]], $[[IMM]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 64-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 64-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @slti3(i32 signext %a) {
entry:
  %cmp = icmp sgt i32 %a, -32770
  %cond = select i1 %cmp, i32 3, i32 5
  ret i32 %cond
}

; 64-bit patterns.

; ALL-LABEL: slti64_0:

; 32-CMOV-DAG:  slt $[[CC:[0-9]+]], $zero, $4
; 32-CMOV-DAG:  addiu $[[I32766:[0-9]+]], $zero, 32766
; 32-CMOV-DAG:  sltu $[[R1:[0-9]+]], $[[I32766]], $5
; 32-CMOV-DAG:  movz $[[CC:[0-9]+]], $[[R1]], $4
; 32-CMOV-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMOV-DAG:  addiu $[[I4:3]], $zero, 4
; 32-CMOV-DAG:  movn $[[I4]], $[[I5]], $[[CC]]
; 32-CMOV-DAG:  addiu $2, $zero, 0

; 32-CMP-DAG:   slt $[[CC0:[0-9]+]], $zero, $4
; 32-CMP-DAG:   addiu $[[I32766:[0-9]+]], $zero, 32766
; 32-CMP-DAG:   sltu $[[CC1:[0-9]+]], $[[I32766]], $5
; 32-CMP-DAG:   selnez $[[CC2:[0-9]+]], $[[CC0]], $4
; 32-CMP-DAG:   seleqz $[[CC3:[0-9]+]], $[[CC1]], $4
; 32-CMP:       or $[[CC:[0-9]+]], $[[CC3]], $[[CC2]]
; 32-CMP-DAG:   addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:   addiu $[[I4:[0-9]+]], $zero, 4
; 32-CMP-DAG:   seleqz $[[T0:[0-9]+]], $[[I4]], $[[CC]]
; 32-CMP-DAG:   selnez $[[T1:[0-9]+]], $[[I5]], $[[CC]]
; 32-CMP-DAG:   or $3, $[[T1]], $[[T0]]
; 32-CMP-DAG:   addiu $2, $zero, 0

; 64-CMOV-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 64-CMOV-DAG:  addiu $[[I4:2]], $zero, 4
; 64-CMOV-DAG:  slti $[[R0:[0-9]+]], $4, 32767
; 64-CMOV-DAG:  movz $[[I4]], $[[I5]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 64-CMP-DAG:  addiu $[[I4:[0-9]+]], $zero, 4
; 64-CMP-DAG:  slti $[[R0:[0-9]+]], $4, 32767
; FIXME: We can do better than this by adding/subtracting the result of slti
;        to/from one of the constants.
; 64-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I4]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i64 @slti64_0(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, 32766
  %conv = select i1 %cmp, i64 5, i64 4
  ret i64 %conv
}

; ALL-LABEL: slti64_1:

; 32-CMOV-DAG:  slt $[[CC:[0-9]+]], $zero, $4
; 32-CMOV-DAG:  addiu $[[I32766:[0-9]+]], $zero, 32767
; 32-CMOV-DAG:  sltu $[[R1:[0-9]+]], $[[I32766]], $5
; 32-CMOV-DAG:  movz $[[CC:[0-9]+]], $[[R1]], $4
; 32-CMOV-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMOV-DAG:  addiu $[[I4:3]], $zero, 4
; 32-CMOV-DAG:  movn $[[I4]], $[[I5]], $[[CC]]
; 32-CMOV-DAG:  addiu $2, $zero, 0

; 32-CMP-DAG:   slt $[[CC0:[0-9]+]], $zero, $4
; 32-CMP-DAG:   addiu $[[I32766:[0-9]+]], $zero, 32767
; 32-CMP-DAG:   sltu $[[CC1:[0-9]+]], $[[I32766]], $5
; 32-CMP-DAG:   selnez $[[CC2:[0-9]+]], $[[CC0]], $4
; 32-CMP-DAG:   seleqz $[[CC3:[0-9]+]], $[[CC1]], $4
; 32-CMP:       or $[[CC:[0-9]+]], $[[CC3]], $[[CC2]]
; 32-CMP-DAG:   addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:   addiu $[[I4:[0-9]+]], $zero, 4
; 32-CMP-DAG:   seleqz $[[T0:[0-9]+]], $[[I4]], $[[CC]]
; 32-CMP-DAG:   selnez $[[T1:[0-9]+]], $[[I5]], $[[CC]]
; 32-CMP-DAG:   or $3, $[[T1]], $[[T0]]
; 32-CMP-DAG:   addiu $2, $zero, 0

; 64-CMOV-DAG: daddiu $[[I5:[0-9]+]], $zero, 5
; 64-CMOV-DAG: daddiu $[[I4:2]], $zero, 4
; 64-CMOV-DAG: daddiu $[[R1:[0-9]+]], $zero, 32767
; 64-CMOV-DAG: slt $[[R0:[0-9]+]], $[[R1]], $4
; 64-CMOV-DAG: movn $[[I4]], $[[I5]], $[[R0]]

; 64-CMP-DAG:  daddiu $[[I5:[0-9]+]], $zero, 5
; 64-CMP-DAG:  daddiu $[[I4:2]], $zero, 4
; 64-CMP-DAG:  daddiu $[[R1:[0-9]+]], $zero, 32767
; 64-CMP-DAG:  slt $[[R0:[0-9]+]], $[[R1]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 64-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I4]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i64 @slti64_1(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, 32767
  %conv = select i1 %cmp, i64 5, i64 4
  ret i64 %conv
}

; ALL-LABEL: slti64_2:

; FIXME: The 32-bit versions of this test are too complicated to reasonably
;        match at the moment. They do show some missing optimizations though
;        such as:
;           (movz $a, $b, (neg $c)) -> (movn $a, $b, $c)

; 64-CMOV-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMOV-DAG:  addiu $[[I4:2]], $zero, 4
; 64-CMOV-DAG:  slti $[[R0:[0-9]+]], $4, -32768
; 64-CMOV-DAG:  movz $[[I4]], $[[I3]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMP-DAG:  addiu $[[I4:[0-9]+]], $zero, 4
; 64-CMP-DAG:  slti $[[R0:[0-9]+]], $4, -32768
; FIXME: We can do better than this by adding/subtracting the result of slti
;        to/from one of the constants.
; 64-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 64-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I4]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i64 @slti64_2(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, -32769
  %conv = select i1 %cmp, i64 3, i64 4
  ret i64 %conv
}

; ALL-LABEL: slti64_3:

; FIXME: The 32-bit versions of this test are too complicated to reasonably
;        match at the moment. They do show some missing optimizations though
;        such as:
;           (movz $a, $b, (neg $c)) -> (movn $a, $b, $c)

; 64-CMOV-DAG: daddiu $[[I5:[0-9]+]], $zero, 5
; 64-CMOV-DAG: daddiu $[[I4:2]], $zero, 4
; 64-CMOV-DAG: daddiu $[[R1:[0-9]+]], ${{[0-9]+}}, 32766
; 64-CMOV-DAG: slt $[[R0:[0-9]+]], $[[R1]], $4
; 64-CMOV-DAG: movn $[[I4]], $[[I5]], $[[R0]]

; 64-CMP-DAG:  daddiu $[[I5:[0-9]+]], $zero, 5
; 64-CMP-DAG:  daddiu $[[I4:2]], $zero, 4
; 64-CMP-DAG:  daddiu $[[R1:[0-9]+]], ${{[0-9]+}}, 32766
; 64-CMP-DAG:  slt $[[R0:[0-9]+]], $[[R1]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 64-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I4]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i64 @slti64_3(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, -32770
  %conv = select i1 %cmp, i64 5, i64 4
  ret i64 %conv
}

; sltiu instructions.

; ALL-LABEL: sltiu0:

; 32-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: sltiu $[[R0:[0-9]+]], $4, 32767
; 32-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  sltiu $[[R0:[0-9]+]], $4, 32767
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 32-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 32-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: sltiu $[[R0:[0-9]+]], $4, 32767
; 64-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 64-CMP-DAG:  sltiu $[[R0:[0-9]+]], $4, 32767
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 64-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 64-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @sltiu0(i32 signext %a) {
entry:
  %cmp = icmp ugt i32 %a, 32766
  %cond = select i1 %cmp, i32 3, i32 5
  ret i32 %cond
}

; ALL-LABEL: sltiu1:

; 32-CMOV-DAG: addiu $[[I7:[0-9]+]], $zero, 7
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: addiu $[[R1:[0-9]+]], $zero, 32767
; 32-CMOV-DAG: sltu $[[R0:[0-9]+]], $[[R1]], $4
; 32-CMOV-DAG: movn $[[I5]], $[[I7]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I7:[0-9]+]], $zero, 7
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  addiu $[[I32767:[0-9]+]], $zero, 32767
; 32-CMP-DAG:  sltu $[[R0:[0-9]+]], $[[I32767]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 32-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I7]], $[[R0]]
; 32-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I7:[0-9]+]], $zero, 7
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: addiu $[[R1:[0-9]+]], $zero, 32767
; 64-CMOV-DAG: sltu $[[R0:[0-9]+]], $[[R1]], $4
; 64-CMOV-DAG: movn $[[I5]], $[[I7]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I7:[0-9]+]], $zero, 7
; 64-CMP-DAG:  addiu $[[I5:2]], $zero, 5
; 64-CMP-DAG:  addiu $[[R1:[0-9]+]], $zero, 32767
; 64-CMP-DAG:  sltu $[[R0:[0-9]+]], $[[R1]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 64-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I7]], $[[R0]]
; 64-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @sltiu1(i32 signext %a) {
entry:
  %cmp = icmp ugt i32 %a, 32767
  %cond = select i1 %cmp, i32 7, i32 5
  ret i32 %cond
}

; ALL-LABEL: sltiu2:

; 32-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: sltiu $[[R0:[0-9]+]], $4, -32768
; 32-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  sltiu $[[R0:[0-9]+]], $4, -32768
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 32-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 32-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: sltiu $[[R0:[0-9]+]], $4, -32768
; 64-CMOV-DAG: movz $[[I5]], $[[I3]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 64-CMP-DAG:  sltiu $[[R0:[0-9]+]], $4, -32768
; FIXME: We can do better than this by using selccz to choose between +0 and +2
; 64-CMP-DAG:  seleqz $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 64-CMP-DAG:  selnez $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @sltiu2(i32 signext %a) {
entry:
  %cmp = icmp ugt i32 %a, -32769
  %cond = select i1 %cmp, i32 3, i32 5
  ret i32 %cond
}

; ALL-LABEL: sltiu3:

; 32-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 32-CMOV-DAG: lui $[[R1:[0-9]+]], 65535
; 32-CMOV-DAG: ori $[[R1]], $[[R1]], 32766
; 32-CMOV-DAG: sltu $[[R0:[0-9]+]], $[[R1]], $4
; 32-CMOV-DAG: movn $[[I5]], $[[I3]], $[[R0]]

; 32-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 32-CMP-DAG:  addiu $[[I5:[0-9]+]], $zero, 5
; 32-CMP-DAG:  lui $[[IMM:[0-9]+]], 65535
; 32-CMP-DAG:  ori $[[IMM]], $[[IMM]], 32766
; 32-CMP-DAG:  sltu $[[R0:[0-9]+]], $[[I32767]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 32-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 32-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 32-CMP-DAG:  or $2, $[[T0]], $[[T1]]

; 64-CMOV-DAG: addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMOV-DAG: addiu $[[I5:2]], $zero, 5
; 64-CMOV-DAG: lui $[[R1:[0-9]+]], 65535
; 64-CMOV-DAG: ori $[[R1]], $[[R1]], 32766
; 64-CMOV-DAG: sltu $[[R0:[0-9]+]], $[[R1]], $4
; 64-CMOV-DAG: movn $[[I5]], $[[I3]], $[[R0]]

; 64-CMP-DAG:  addiu $[[I3:[0-9]+]], $zero, 3
; 64-CMP-DAG:  addiu $[[I5:2]], $zero, 5
; 64-CMP-DAG:  lui $[[IMM:[0-9]+]], 65535
; 64-CMP-DAG:  ori $[[IMM]], $[[IMM]], 32766
; 64-CMP-DAG:  sltu $[[R0:[0-9]+]], $[[IMM]], $4
; FIXME: We can do better than this by using selccz to choose between -0 and -2
; 64-CMP-DAG:  selnez $[[T0:[0-9]+]], $[[I3]], $[[R0]]
; 64-CMP-DAG:  seleqz $[[T1:[0-9]+]], $[[I5]], $[[R0]]
; 64-CMP-DAG:  or $2, $[[T0]], $[[T1]]

define i32 @sltiu3(i32 signext %a) {
entry:
  %cmp = icmp ugt i32 %a, -32770
  %cond = select i1 %cmp, i32 3, i32 5
  ret i32 %cond
}

; Check if
;  (select (setxx a, N), x, x-1) or
;  (select (setxx a, N), x-1, x)
; doesn't generate conditional moves
; for constant operands whose difference is |1|

define i32 @slti4(i32 signext %a) nounwind readnone {
  %1 = icmp slt i32 %a, 7
  %2 = select i1 %1, i32 4, i32 3
  ret i32 %2
}

; ALL-LABEL: slti4:

; 32-CMOV-DAG: slti [[R1:\$[0-9]+]], $4, 7
; 32-CMOV-DAG: addiu $2, [[R1]], 3
; 32-CMOV-NOT: movn

; 32-CMP-DAG:  slti [[R1:\$[0-9]+]], $4, 7
; 32-CMP-DAG:  addiu $2, [[R1]], 3
; 32-CMP-NOT:  seleqz
; 32-CMP-NOT:  selnez

; 64-CMOV-DAG: slti [[R1:\$[0-9]+]], $4, 7
; 64-CMOV-DAG: addiu $2, [[R1]], 3
; 64-CMOV-NOT: movn

; 64-CMP-DAG:  slti [[R1:\$[0-9]+]], $4, 7
; 64-CMP-DAG:  addiu $2, [[R1]], 3
; 64-CMP-NOT:  seleqz
; 64-CMP-NOT:  selnez

define i32 @slti5(i32 signext %a) nounwind readnone {
  %1 = icmp slt i32 %a, 7
  %2 = select i1 %1, i32 -3, i32 -4
  ret i32 %2
}

; ALL-LABEL: slti5:

; 32-CMOV-DAG: slti [[R1:\$[0-9]+]], $4, 7
; 32-CMOV-DAG: addiu [[R3:\$[0-9]+]], [[R2:\$[a-z0-9]+]], -4
; 32-CMOV-NOT: movn

; 32-CMP-DAG:  slti [[R1:\$[0-9]+]], $4, 7
; 32-CMP-DAG:  addiu [[R3:\$[0-9]+]], [[R2:\$[a-z0-9]+]], -4
; 32-CMP-NOT:  seleqz
; 32-CMP-NOT:  selnez

; 64-CMOV-DAG: slti [[R1:\$[0-9]+]], $4, 7
; 64-CMOV-DAG: addiu [[R3:\$[0-9]+]], [[R2:\$[a-z0-9]+]], -4
; 64-CMOV-NOT: movn

; 64-CMP-DAG:  slti [[R1:\$[0-9]+]], $4, 7
; 64-CMP-DAG:  addiu [[R3:\$[0-9]+]], [[R2:\$[a-z0-9]+]], -4
; 64-CMP-NOT:  seleqz
; 64-CMP-NOT:  selnez

define i32 @slti6(i32 signext %a) nounwind readnone {
  %1 = icmp slt i32 %a, 7
  %2 = select i1 %1, i32 3, i32 4
  ret i32 %2
}

; ALL-LABEL: slti6:

; ALL-DAG: addiu [[R1:\$[0-9]+]], $zero, 6
; ALL-DAG: slt [[R1]], [[R1]], $4
; ALL-DAG: addiu [[R2:\$[0-9]+]], [[R1]], 3
; ALL-NOT: movn
; ALL-NOT:  seleqz
; ALL-NOT:  selnez
