; RUN: llc -march=mipsel --disable-machine-licm -mcpu=mips32   -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS32-ANY,NO-SEB-SEH,CHECK-EL,NOT-MICROMIPS
; RUN: llc -march=mipsel --disable-machine-licm -mcpu=mips32r2 -relocation-model=pic -verify-machineinstrs < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS32-ANY,HAS-SEB-SEH,CHECK-EL,NOT-MICROMIPS
; RUN: llc -march=mipsel --disable-machine-licm -mcpu=mips32r6 -relocation-model=pic -verify-machineinstrs < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS32-ANY,HAS-SEB-SEH,CHECK-EL,MIPSR6
; RUN: llc -march=mips64el --disable-machine-licm -mcpu=mips4    -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS64-ANY,NO-SEB-SEH,CHECK-EL,NOT-MICROMIPS
; RUN: llc -march=mips64el --disable-machine-licm -mcpu=mips64   -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS64-ANY,NO-SEB-SEH,CHECK-EL,NOT-MICROMIPS
; RUN: llc -march=mips64el --disable-machine-licm -mcpu=mips64r2 -relocation-model=pic -verify-machineinstrs < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS64-ANY,HAS-SEB-SEH,CHECK-EL,NOT-MICROMIPS
; RUN: llc -march=mips64el --disable-machine-licm -mcpu=mips64r6 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS64-ANY,HAS-SEB-SEH,CHECK-EL,MIPSR6
; RUN: llc -march=mips64 -O0 -mcpu=mips64r6 -relocation-model=pic -verify-machineinstrs < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL-LABEL,MIPS64-ANY,O0
; RUN: llc -march=mipsel --disable-machine-licm -mcpu=mips32r2 -mattr=micromips -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS32-ANY,HAS-SEB-SEH,CHECK-EL,MICROMIPS

; Keep one big-endian check so that we don't reduce testing, but don't add more
; since endianness doesn't affect the body of the atomic operations.
; RUN: llc -march=mips   --disable-machine-licm -mcpu=mips32 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MIPS32-ANY,NO-SEB-SEH,CHECK-EB,NOT-MICROMIPS

@x = common global i32 0, align 4

define i32 @AtomicLoadAdd32(i32 signext %incr) nounwind {
entry:
  %0 = atomicrmw add i32* @x, i32 %incr monotonic
  ret i32 %0

; ALL-LABEL: AtomicLoadAdd32:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(x)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(x)(

; O0:        [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; O0:            ld      $[[R1:[0-9]+]]
; O0-NEXT:       ll      $[[R2:[0-9]+]], 0($[[R1]])

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R3:[0-9]+]], 0($[[R0]])
; ALL:           addu    $[[R4:[0-9]+]], $[[R3]], $4
; ALL:           sc      $[[R4]], 0($[[R0]])
; NOT-MICROMIPS: beqz    $[[R4]], [[BB0]]
; MICROMIPS:     beqzc   $[[R4]], [[BB0]]
; MIPSR6:        beqzc   $[[R4]], [[BB0]]
}

define i32 @AtomicLoadNand32(i32 signext %incr) nounwind {
entry:
  %0 = atomicrmw nand i32* @x, i32 %incr monotonic
  ret i32 %0

; ALL-LABEL: AtomicLoadNand32:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(x)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(x)(



; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R1:[0-9]+]], 0($[[R0]])
; ALL:           and     $[[R3:[0-9]+]], $[[R1]], $4
; ALL:           nor     $[[R2:[0-9]+]], $zero, $[[R3]]
; ALL:           sc      $[[R2]], 0($[[R0]])
; NOT-MICROMIPS: beqz    $[[R2]], [[BB0]]
; MICROMIPS:     beqzc   $[[R2]], [[BB0]]
; MIPSR6:        beqzc   $[[R2]], [[BB0]]
}

define i32 @AtomicSwap32(i32 signext %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32, i32* %newval.addr, align 4
  %0 = atomicrmw xchg i32* @x, i32 %tmp monotonic
  ret i32 %0

; ALL-LABEL: AtomicSwap32:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(x)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(x)

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      ${{[0-9]+}}, 0($[[R0]])
; ALL:           sc      $[[R2:[0-9]+]], 0($[[R0]])
; NOT-MICROMIPS: beqz    $[[R2]], [[BB0]]
; MICROMIPS:     beqzc   $[[R2]], [[BB0]]
; MIPSR6:        beqzc   $[[R2]], [[BB0]]
}

define i32 @AtomicCmpSwap32(i32 signext %oldval, i32 signext %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32, i32* %newval.addr, align 4
  %0 = cmpxchg i32* @x, i32 %oldval, i32 %tmp monotonic monotonic
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1

; ALL-LABEL: AtomicCmpSwap32:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(x)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(x)(

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $2, 0($[[R0]])
; NOT-MICROMIPS: bne     $2, $4, [[BB1:(\$|\.L)[A-Z_0-9]+]]
; MICROMIPS:     bne     $2, $4, [[BB1:(\$|\.L)[A-Z_0-9]+]]
; MIPSR6:        bnec    $2, $4, [[BB1:(\$|\.L)[A-Z_0-9]+]]
; ALL:           sc      $[[R2:[0-9]+]], 0($[[R0]])
; NOT-MICROMIPS: beqz    $[[R2]], [[BB0]]
; MICROMIPS:     beqzc   $[[R2]], [[BB0]]
; MIPSR6:        beqzc   $[[R2]], [[BB0]]
; ALL:       [[BB1]]:
}



@y = common global i8 0, align 1

define signext i8 @AtomicLoadAdd8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw add i8* @y, i8 %incr monotonic
  ret i8 %0

; ALL-LABEL: AtomicLoadAdd8:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(y)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(y)(

; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; O0:        [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; O0:            ld      $[[R10:[0-9]+]]
; O0-NEXT:       ll      $[[R11:[0-9]+]], 0($[[R10]])

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R12:[0-9]+]], 0($[[R2]])
; ALL:           addu    $[[R13:[0-9]+]], $[[R12]], $[[R9]]
; ALL:           and     $[[R14:[0-9]+]], $[[R13]], $[[R7]]
; ALL:           and     $[[R15:[0-9]+]], $[[R12]], $[[R8]]
; ALL:           or      $[[R16:[0-9]+]], $[[R15]], $[[R14]]
; ALL:           sc      $[[R16]], 0($[[R2]])
; NOT-MICROMIPS: beqz    $[[R16]], [[BB0]]
; MICROMIPS:     beqzc   $[[R16]], [[BB0]]
; MIPSR6:        beqzc   $[[R16]], [[BB0]]

; ALL:           and     $[[R17:[0-9]+]], $[[R12]], $[[R7]]
; ALL:           srlv    $[[R18:[0-9]+]], $[[R17]], $[[R5]]

; NO-SEB-SEH:    sll     $[[R19:[0-9]+]], $[[R18]], 24
; NO-SEB-SEH:    sra     $2, $[[R19]], 24

; HAS-SEB-SEH:   seb     $2, $[[R18]]
}

define signext i8 @AtomicLoadSub8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw sub i8* @y, i8 %incr monotonic
  ret i8 %0

; ALL-LABEL: AtomicLoadSub8:

; MIPS32-ANY: lw      $[[R0:[0-9]+]], %got(y)
; MIPS64-ANY: ld      $[[R0:[0-9]+]], %got_disp(y)(

; ALL:        addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:        and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:        andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:   sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:   xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:   sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:        ori     $[[R6:[0-9]+]], $zero, 255
; ALL:        sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:        nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:        sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; O0:        [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; O0:            ld      $[[R10:[0-9]+]]
; O0-NEXT:       ll      $[[R11:[0-9]+]], 0($[[R10]])

; ALL:    [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:        ll      $[[R12:[0-9]+]], 0($[[R2]])
; ALL:        subu    $[[R13:[0-9]+]], $[[R12]], $[[R9]]
; ALL:        and     $[[R14:[0-9]+]], $[[R13]], $[[R7]]
; ALL:        and     $[[R15:[0-9]+]], $[[R12]], $[[R8]]
; ALL:        or      $[[R16:[0-9]+]], $[[R15]], $[[R14]]
; ALL:        sc      $[[R16]], 0($[[R2]])
; NOT-MICROMIPS: beqz    $[[R16]], [[BB0]]
; MICROMIPS:  beqzc   $[[R16]], [[BB0]]
; MIPSR6:     beqzc   $[[R16]], [[BB0]]

; ALL:        and     $[[R17:[0-9]+]], $[[R12]], $[[R7]]
; ALL:        srlv    $[[R18:[0-9]+]], $[[R17]], $[[R5]]

; NO-SEB-SEH: sll     $[[R19:[0-9]+]], $[[R18]], 24
; NO-SEB-SEH: sra     $2, $[[R19]], 24

; HAS-SEB-SEH:seb     $2, $[[R18]]
}

define signext i8 @AtomicLoadNand8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw nand i8* @y, i8 %incr monotonic
  ret i8 %0

; ALL-LABEL: AtomicLoadNand8:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(y)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(y)(

; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; O0:        [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; O0:            ld      $[[R10:[0-9]+]]
; O0-NEXT:       ll      $[[R11:[0-9]+]], 0($[[R10]])

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R12:[0-9]+]], 0($[[R2]])
; ALL:           and     $[[R13:[0-9]+]], $[[R12]], $[[R9]]
; ALL:           nor     $[[R14:[0-9]+]], $zero, $[[R13]]
; ALL:           and     $[[R15:[0-9]+]], $[[R14]], $[[R7]]
; ALL:           and     $[[R16:[0-9]+]], $[[R12]], $[[R8]]
; ALL:           or      $[[R17:[0-9]+]], $[[R16]], $[[R15]]
; ALL:           sc      $[[R17]], 0($[[R2]])
; NOT-MICROMIPS: beqz    $[[R17]], [[BB0]]
; MICROMIPS:     beqzc   $[[R17]], [[BB0]]
; MIPSR6:        beqzc   $[[R17]], [[BB0]]

; ALL:           and     $[[R18:[0-9]+]], $[[R12]], $[[R7]]
; ALL:           srlv    $[[R19:[0-9]+]], $[[R18]], $[[R5]]

; NO-SEB-SEH:    sll     $[[R20:[0-9]+]], $[[R19]], 24
; NO-SEB-SEH:    sra     $2, $[[R20]], 24

; HAS-SEB-SEH:   seb     $2, $[[R19]]
}

define signext i8 @AtomicSwap8(i8 signext %newval) nounwind {
entry:
  %0 = atomicrmw xchg i8* @y, i8 %newval monotonic
  ret i8 %0

; ALL-LABEL: AtomicSwap8:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(y)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(y)(

; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R10:[0-9]+]], 0($[[R2]])
; ALL:           and     $[[R18:[0-9]+]], $[[R9]], $[[R7]]
; ALL:           and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; ALL:           or      $[[R14:[0-9]+]], $[[R13]], $[[R18]]
; ALL:           sc      $[[R14]], 0($[[R2]])
; NOT-MICROMIPS: beqz    $[[R14]], [[BB0]]
; MICROMIPS:     beqzc   $[[R14]], [[BB0]]
; MIPSR6:        beqzc   $[[R14]], [[BB0]]

; ALL:           and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; ALL:           srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]

; NO-SEB-SEH:    sll     $[[R17:[0-9]+]], $[[R16]], 24
; NO-SEB-SEH:    sra     $2, $[[R17]], 24

; HAS-SEB-SEH:   seb     $2, $[[R16]]

}

define signext i8 @AtomicCmpSwap8(i8 signext %oldval, i8 signext %newval) nounwind {
entry:
  %pair0 = cmpxchg i8* @y, i8 %oldval, i8 %newval monotonic monotonic
  %0 = extractvalue { i8, i1 } %pair0, 0
  ret i8 %0

; ALL-LABEL: AtomicCmpSwap8:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(y)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(y)(

; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           andi    $[[R9:[0-9]+]], $4, 255
; ALL:           sllv    $[[R10:[0-9]+]], $[[R9]], $[[R5]]
; ALL:           andi    $[[R11:[0-9]+]], $5, 255
; ALL:           sllv    $[[R12:[0-9]+]], $[[R11]], $[[R5]]

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R13:[0-9]+]], 0($[[R2]])
; ALL:           and     $[[R14:[0-9]+]], $[[R13]], $[[R7]]
; NOT-MICROMIPS: bne     $[[R14]], $[[R10]], [[BB1:(\$|\.L)[A-Z_0-9]+]]
; MICROMIPS:     bne     $[[R14]], $[[R10]], [[BB1:(\$|\.L)[A-Z_0-9]+]]
; MIPSR6:        bnec    $[[R14]], $[[R10]], [[BB1:(\$|\.L)[A-Z_0-9]+]]

; ALL:           and     $[[R15:[0-9]+]], $[[R13]], $[[R8]]
; ALL:           or      $[[R16:[0-9]+]], $[[R15]], $[[R12]]
; ALL:           sc      $[[R16]], 0($[[R2]])
; NOT-MICROMIPS: beqz    $[[R16]], [[BB0]]
; MICROMIPS:     beqzc   $[[R16]], [[BB0]]
; MIPSR6:        beqzc   $[[R16]], [[BB0]]

; ALL:       [[BB1]]:
; ALL:           srlv    $[[R17:[0-9]+]], $[[R14]], $[[R5]]

; NO-SEB-SEH:    sll     $[[R18:[0-9]+]], $[[R17]], 24
; NO-SEB-SEH:    sra     $2, $[[R18]], 24

; HAS-SEB-SEH:   seb     $2, $[[R17]]
}

define i1 @AtomicCmpSwapRes8(i8* %ptr, i8 signext %oldval, i8 signext %newval) nounwind {
entry:
  %0 = cmpxchg i8* %ptr, i8 %oldval, i8 %newval monotonic monotonic
  %1 = extractvalue { i8, i1 } %0, 1
  ret i1 %1
; ALL-LABEL: AtomicCmpSwapRes8

; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $4, $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $4, 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           andi    $[[R9:[0-9]+]], $5, 255
; ALL:           sllv    $[[R10:[0-9]+]], $[[R9]], $[[R5]]
; ALL:           andi    $[[R11:[0-9]+]], $6, 255
; ALL:           sllv    $[[R12:[0-9]+]], $[[R11]], $[[R5]]

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R13:[0-9]+]], 0($[[R2]])
; ALL:           and     $[[R14:[0-9]+]], $[[R13]], $[[R7]]
; NOT-MICROMIPS: bne     $[[R14]], $[[R10]], [[BB1:(\$|\.L)[A-Z_0-9]+]]
; MICROMIPS:     bne     $[[R14]], $[[R10]], [[BB1:(\$|\.L)[A-Z_0-9]+]]
; MIPSR6:        bnec    $[[R14]], $[[R10]], [[BB1:(\$|\.L)[A-Z_0-9]+]]

; ALL:           and     $[[R15:[0-9]+]], $[[R13]], $[[R8]]
; ALL:           or      $[[R16:[0-9]+]], $[[R15]], $[[R12]]
; ALL:           sc      $[[R16]], 0($[[R2]])
; NOT-MICROMIPS: beqz    $[[R16]], [[BB0]]
; MICROMIPS:     beqzc   $[[R16]], [[BB0]]
; MIPSR6:        beqzc   $[[R16]], [[BB0]]

; ALL:       [[BB1]]:
; ALL:           srlv    $[[R17:[0-9]+]], $[[R14]], $[[R5]]

; NO-SEB-SEH:    sll     $[[R18:[0-9]+]], $[[R17]], 24
; NO-SEB-SEH:    sra     $[[R19:[0-9]+]], $[[R18]], 24

; FIXME: -march=mips produces a redundant sign extension here...
; NO-SEB-SEH:    sll     $[[R20:[0-9]+]], $5, 24
; NO-SEB-SEH:    sra     $[[R20]], $[[R20]], 24

; HAS-SEB-SEH:   seb     $[[R19:[0-9]+]], $[[R17]]

; FIXME: ...Leading to this split check.
; NO-SEB-SEH:    xor     $[[R21:[0-9]+]], $[[R19]], $[[R20]]
; HAS-SEB-SEH:   xor     $[[R21:[0-9]+]], $[[R19]], $5

; ALL: sltiu   $2, $[[R21]], 1
}

; Check one i16 so that we cover the seh sign extend
@z = common global i16 0, align 1

define signext i16 @AtomicLoadAdd16(i16 signext %incr) nounwind {
entry:
  %0 = atomicrmw add i16* @z, i16 %incr monotonic
  ret i16 %0

; ALL-LABEL: AtomicLoadAdd16:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(z)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(z)(

; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 2
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 65535
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; O0:        [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; O0:            ld      $[[R10:[0-9]+]]
; O0-NEXT:       ll      $[[R11:[0-9]+]], 0($[[R10]])

; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R12:[0-9]+]], 0($[[R2]])
; ALL:           addu    $[[R13:[0-9]+]], $[[R12]], $[[R9]]
; ALL:           and     $[[R14:[0-9]+]], $[[R13]], $[[R7]]
; ALL:           and     $[[R15:[0-9]+]], $[[R12]], $[[R8]]
; ALL:           or      $[[R16:[0-9]+]], $[[R15]], $[[R14]]
; ALL:           sc      $[[R16]], 0($[[R2]])
; NOT-MICROMIPS: beqz    $[[R16]], [[BB0]]
; MICROMIPS:     beqzc   $[[R16]], [[BB0]]
; MIPSR6:        beqzc   $[[R16]], [[BB0]]

; ALL:           and     $[[R17:[0-9]+]], $[[R12]], $[[R7]]
; ALL:           srlv    $[[R18:[0-9]+]], $[[R17]], $[[R5]]

; NO-SEB-SEH:    sll     $[[R19:[0-9]+]], $[[R18]], 16
; NO-SEB-SEH:    sra     $2, $[[R19]], 16

; MIPS32R2:      seh     $2, $[[R18]]
}

; Test that the i16 return value from cmpxchg is recognised as signed,
; so that setCC doesn't end up comparing an unsigned value to a signed
; value.
; The rest of the functions here are testing the atomic expansion, so
; we just match the end of the function.
define {i16, i1} @foo(i16* %addr, i16 %l, i16 %r, i16 %new) {
  %desired = add i16 %l, %r
  %res = cmpxchg i16* %addr, i16 %desired, i16 %new seq_cst seq_cst
  ret {i16, i1} %res

; ALL-LABEL: foo
; MIPSR6:        addu    $[[R2:[0-9]+]], $[[R1:[0-9]+]], $[[R0:[0-9]+]]
; NOT-MICROMIPS: addu    $[[R2:[0-9]+]], $[[R1:[0-9]+]], $[[R0:[0-9]+]]
; MICROMIPS:     addu16  $[[R2:[0-9]+]], $[[R1:[0-9]+]], $[[R0:[0-9]+]]

; ALL:           sync

; ALL:           andi    $[[R3:[0-9]+]], $[[R2]], 65535
; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R4:[0-9]+]], 0($[[R5:[0-9]+]])
; ALL:           and     $[[R6:[0-9]+]], $[[R4]], $
; ALL:           and     $[[R7:[0-9]+]], $[[R4]], $
; ALL:           or      $[[R8:[0-9]+]], $[[R7]], $
; ALL:           sc      $[[R8]], 0($[[R5]])
; NOT-MICROMIPS: beqz    $[[R8]], [[BB0]]
; MICROMIPS:     beqzc   $[[R8]], [[BB0]]
; MIPSR6:        beqzc   $[[R8]], [[BB0]]

; ALL:           srlv    $[[R9:[0-9]+]], $[[R6]], $

; NO-SEB-SEH:    sll     $[[R10:[0-9]+]], $[[R9]], 16
; NO-SEB-SEH:    sra     $[[R11:[0-9]+]], $[[R10]], 16

; NO-SEB-SEH:    sll     $[[R12:[0-9]+]], $[[R2]], 16
; NO-SEB-SEH:    sra     $[[R13:[0-9]+]], $[[R12]], 16

; HAS-SEB-SEH:   seh     $[[R11:[0-9]+]], $[[R9]]
; HAS-SEB-SEH:   seh     $[[R13:[0-9]+]], $[[R2]]

; ALL:           xor     $[[R12:[0-9]+]], $[[R11]], $[[R13]]
; ALL:           sltiu   $3, $[[R12]], 1
; ALL:           sync
}

@countsint = common global i32 0, align 4

define i32 @CheckSync(i32 signext %v) nounwind noinline {
entry:
  %0 = atomicrmw add i32* @countsint, i32 %v seq_cst
  ret i32 %0 

; ALL-LABEL: CheckSync:

; ALL:           sync
; ALL:           ll
; ALL:           sc
; ALL:           beq
; ALL:           sync
}

; make sure that this assertion in
; TwoAddressInstructionPass::TryInstructionTransform does not fail:
;
; line 1203: assert(TargetRegisterInfo::isVirtualRegister(regB) &&
;
; it failed when MipsDAGToDAGISel::ReplaceUsesWithZeroReg replaced an
; operand of an atomic instruction with register $zero. 
@a = external global i32

define i32 @zeroreg() nounwind {
entry:
  %pair0 = cmpxchg i32* @a, i32 1, i32 0 seq_cst seq_cst
  %0 = extractvalue { i32, i1 } %pair0, 0
  %1 = icmp eq i32 %0, 1
  %conv = zext i1 %1 to i32
  ret i32 %conv
}

; Check that MIPS32R6 has the correct offset range.
; FIXME: At the moment, we don't seem to do addr+offset for any atomic load/store.
define i32 @AtomicLoadAdd32_OffGt9Bit(i32 signext %incr) nounwind {
entry:
  %0 = atomicrmw add i32* getelementptr(i32, i32* @x, i32 256), i32 %incr monotonic
  ret i32 %0

; ALL-LABEL: AtomicLoadAdd32_OffGt9Bit:

; MIPS32-ANY:    lw      $[[R0:[0-9]+]], %got(x)
; MIPS64-ANY:    ld      $[[R0:[0-9]+]], %got_disp(x)(

; ALL:           addiu   $[[PTR:[0-9]+]], $[[R0]], 1024
; ALL:       [[BB0:(\$|\.L)[A-Z_0-9]+]]:
; ALL:           ll      $[[R1:[0-9]+]], 0($[[PTR]])
; ALL:           addu    $[[R2:[0-9]+]], $[[R1]], $4
; ALL:           sc      $[[R2]], 0($[[PTR]])
; NOT-MICROMIPS: beqz    $[[R2]], [[BB0]]
; MICROMIPS:     beqzc   $[[R2]], [[BB0]]
; MIPSR6:        beqzc   $[[R2]], [[BB0]]
}
