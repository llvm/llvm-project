; RUN: llc -march=mipsel < %s | FileCheck %s
; RUN: llc -march=mipsel -force-mips-long-branch -O3 < %s \
; RUN:   | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips64el -mcpu=mips4 -mattr=n64 -force-mips-long-branch -O3 \
; RUN:   < %s | FileCheck %s -check-prefix=N64
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=n64 -force-mips-long-branch -O3 \
; RUN:   < %s | FileCheck %s -check-prefix=N64
; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=micromips \
; RUN:   -force-mips-long-branch -O3 < %s | FileCheck %s -check-prefix=MICROMIPS
; RUN: llc -mtriple=mipsel-none-nacl -force-mips-long-branch -O3 < %s \
; RUN:   | FileCheck %s -check-prefix=NACL


@x = external global i32

define void @test1(i32 signext %s) {
entry:
  %cmp = icmp eq i32 %s, 0
  br i1 %cmp, label %end, label %then

then:
  store i32 1, i32* @x, align 4
  br label %end

end:
  ret void


; First check the normal version (without long branch).  beqz jumps to return,
; and fallthrough block stores 1 to global variable.

; CHECK:        lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[R0]], $[[R0]], %lo(_gp_disp)
; CHECK:        beqz    $4, $[[BB0:BB[0-9_]+]]
; CHECK:        addu    $[[GP:[0-9]+]], $[[R0]], $25
; CHECK:        lw      $[[R1:[0-9]+]], %got(x)($[[GP]])
; CHECK:        addiu   $[[R2:[0-9]+]], $zero, 1
; CHECK:        sw      $[[R2]], 0($[[R1]])
; CHECK:   $[[BB0]]:
; CHECK:        jr      $ra
; CHECK:        nop


; Check the MIPS32 version.  Check that branch logic is inverted, so that the
; target of the new branch (bnez) is the fallthrough block of the original
; branch.  Check that fallthrough block of the new branch contains long branch
; expansion which at the end indirectly jumps to the target of the original
; branch.

; O32:        lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; O32:        addiu   $[[R0]], $[[R0]], %lo(_gp_disp)
; O32:        bnez    $4, $[[BB0:BB[0-9_]+]]
; O32:        addu    $[[GP:[0-9]+]], $[[R0]], $25

; Check for long branch expansion:
; O32:             addiu   $sp, $sp, -8
; O32-NEXT:        sw      $ra, 0($sp)
; O32-NEXT:        lui     $1, %hi(($[[BB2:BB[0-9_]+]])-($[[BB1:BB[0-9_]+]]))
; O32-NEXT:        bal     $[[BB1]]
; O32-NEXT:        addiu   $1, $1, %lo(($[[BB2]])-($[[BB1]]))
; O32-NEXT:   $[[BB1]]:
; O32-NEXT:        addu    $1, $ra, $1
; O32-NEXT:        lw      $ra, 0($sp)
; O32-NEXT:        jr      $1
; O32-NEXT:        addiu   $sp, $sp, 8

; O32:   $[[BB0]]:
; O32:        lw      $[[R1:[0-9]+]], %got(x)($[[GP]])
; O32:        addiu   $[[R2:[0-9]+]], $zero, 1
; O32:        sw      $[[R2]], 0($[[R1]])
; O32:   $[[BB2]]:
; O32:        jr      $ra
; O32:        nop


; Check the MIPS64 version.

; N64:        lui     $[[R0:[0-9]+]], %hi(%neg(%gp_rel(test1)))
; N64:        bnez    $4, $[[BB0:BB[0-9_]+]]
; N64:        daddu   $[[R1:[0-9]+]], $[[R0]], $25

; Check for long branch expansion:
; N64:           daddiu  $sp, $sp, -16
; N64-NEXT:      sd      $ra, 0($sp)
; N64-NEXT:      daddiu  $1, $zero, %hi(($[[BB2:BB[0-9_]+]])-($[[BB1:BB[0-9_]+]]))
; N64-NEXT:      dsll    $1, $1, 16
; N64-NEXT:      bal     $[[BB1]]
; N64-NEXT:      daddiu  $1, $1, %lo(($[[BB2]])-($[[BB1]]))
; N64-NEXT:  $[[BB1]]:
; N64-NEXT:      daddu   $1, $ra, $1
; N64-NEXT:      ld      $ra, 0($sp)
; N64-NEXT:      jr      $1
; N64-NEXT:      daddiu  $sp, $sp, 16

; N64:   $[[BB0]]:
; N64:        daddiu  $[[GP:[0-9]+]], $[[R1]], %lo(%neg(%gp_rel(test1)))
; N64:        ld      $[[R2:[0-9]+]], %got_disp(x)($[[GP]])
; N64:        addiu   $[[R3:[0-9]+]], $zero, 1
; N64:        sw      $[[R3]], 0($[[R2]])
; N64:   $[[BB2]]:
; N64:        jr      $ra
; N64:        nop


; Check the microMIPS version.

; MICROMIPS:        lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; MICROMIPS:        addiu   $[[R0]], $[[R0]], %lo(_gp_disp)
; MICROMIPS:        bnez    $4, $[[BB0:BB[0-9_]+]]
; MICROMIPS:        addu    $[[GP:[0-9]+]], $[[R0]], $25

; Check for long branch expansion:
; MICROMIPS:          addiu   $sp, $sp, -8
; MICROMIPS-NEXT:     sw      $ra, 0($sp)
; MICROMIPS-NEXT:     lui     $1, %hi(($[[BB2:BB[0-9_]+]])-($[[BB1:BB[0-9_]+]]))
; MICROMIPS-NEXT:     bal     $[[BB1]]
; MICROMIPS-NEXT:     addiu   $1, $1, %lo(($[[BB2]])-($[[BB1]]))
; MICROMIPS-NEXT:  $[[BB1]]:
; MICROMIPS-NEXT:     addu    $1, $ra, $1
; MICROMIPS-NEXT:     lw      $ra, 0($sp)
; MICROMIPS-NEXT:     jr      $1
; MICROMIPS-NEXT:     addiu   $sp, $sp, 8

; MICROMIPS:   $[[BB0]]:
; MICROMIPS:        lw      $[[R1:[0-9]+]], %got(x)($[[GP]])
; MICROMIPS:        addiu   $[[R2:[0-9]+]], $zero, 1
; MICROMIPS:        sw      $[[R2]], 0($[[R1]])
; MICROMIPS:   $[[BB2]]:
; MICROMIPS:        jr      $ra
; MICROMIPS:        nop


; Check the NaCl version.  Check that sp change is not in the branch delay slot
; of "jr $1" instruction.  Check that target of indirect branch "jr $1" is
; bundle aligned.

; NACL:        lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; NACL:        addiu   $[[R0]], $[[R0]], %lo(_gp_disp)
; NACL:        bnez    $4, $[[BB0:BB[0-9_]+]]
; NACL:        addu    $[[GP:[0-9]+]], $[[R0]], $25

; Check for long branch expansion:
; NACL:             addiu   $sp, $sp, -8
; NACL-NEXT:        sw      $ra, 0($sp)
; NACL-NEXT:        lui     $1, %hi(($[[BB2:BB[0-9_]+]])-($[[BB1:BB[0-9_]+]]))
; NACL-NEXT:        bal     $[[BB1]]
; NACL-NEXT:        addiu   $1, $1, %lo(($[[BB2]])-($[[BB1]]))
; NACL-NEXT:   $[[BB1]]:
; NACL-NEXT:        addu    $1, $ra, $1
; NACL-NEXT:        lw      $ra, 0($sp)
; NACL-NEXT:        addiu   $sp, $sp, 8
; NACL-NEXT:        jr      $1
; NACL-NEXT:        nop

; NACL:        $[[BB0]]:
; NACL:             lw      $[[R1:[0-9]+]], %got(x)($[[GP]])
; NACL:             addiu   $[[R2:[0-9]+]], $zero, 1
; NACL:             sw      $[[R2]], 0($[[R1]])
; NACL:             .align  4
; NACL-NEXT:   $[[BB2]]:
; NACL:             jr      $ra
; NACL:             nop
}
