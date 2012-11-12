; RUN: llc -march=mipsel --disable-machine-licm < %s | FileCheck %s

@x = common global i32 0, align 4

define i32 @AtomicLoadAdd32(i32 %incr) nounwind {
entry:
  %0 = atomicrmw add i32* @x, i32 %incr monotonic
  ret i32 %0

; CHECK:   AtomicLoadAdd32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK:   addu    $[[R2:[0-9]+]], $[[R1]], $4
; CHECK:   sc      $[[R2]], 0($[[R0]])
; CHECK:   beq     $[[R2]], $zero, $[[BB0]]
}

define i32 @AtomicLoadNand32(i32 %incr) nounwind {
entry:
  %0 = atomicrmw nand i32* @x, i32 %incr monotonic
  ret i32 %0

; CHECK:   AtomicLoadNand32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK:   and     $[[R3:[0-9]+]], $[[R1]], $4
; CHECK:   nor     $[[R2:[0-9]+]], $zero, $[[R3]]
; CHECK:   sc      $[[R2]], 0($[[R0]])
; CHECK:   beq     $[[R2]], $zero, $[[BB0]]
}

define i32 @AtomicSwap32(i32 %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32* %newval.addr, align 4
  %0 = atomicrmw xchg i32* @x, i32 %tmp monotonic
  ret i32 %0

; CHECK:   AtomicSwap32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      ${{[0-9]+}}, 0($[[R0]])
; CHECK:   sc      $[[R2:[0-9]+]], 0($[[R0]])
; CHECK:   beq     $[[R2]], $zero, $[[BB0]]
}

define i32 @AtomicCmpSwap32(i32 %oldval, i32 %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32* %newval.addr, align 4
  %0 = cmpxchg i32* @x, i32 %oldval, i32 %tmp monotonic
  ret i32 %0

; CHECK:   AtomicCmpSwap32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $2, 0($[[R0]])
; CHECK:   bne     $2, $4, $[[BB1:[A-Z_0-9]+]]
; CHECK:   sc      $[[R2:[0-9]+]], 0($[[R0]])
; CHECK:   beq     $[[R2]], $zero, $[[BB0]]
; CHECK:   $[[BB1]]:
}



@y = common global i8 0, align 1

define signext i8 @AtomicLoadAdd8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw add i8* @y, i8 %incr monotonic
  ret i8 %0

; CHECK:   AtomicLoadAdd8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   sllv    $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   addu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadSub8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw sub i8* @y, i8 %incr monotonic
  ret i8 %0

; CHECK:   AtomicLoadSub8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   sllv     $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   subu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadNand8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw nand i8* @y, i8 %incr monotonic
  ret i8 %0

; CHECK:   AtomicLoadNand8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   sllv    $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   and     $[[R18:[0-9]+]], $[[R10]], $[[R9]]
; CHECK:   nor     $[[R11:[0-9]+]], $zero, $[[R18]]
; CHECK:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicSwap8(i8 signext %newval) nounwind {
entry:
  %0 = atomicrmw xchg i8* @y, i8 %newval monotonic
  ret i8 %0

; CHECK:   AtomicSwap8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   sllv    $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   and     $[[R18:[0-9]+]], $[[R9]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R18]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicCmpSwap8(i8 signext %oldval, i8 signext %newval) nounwind {
entry:
  %0 = cmpxchg i8* @y, i8 %oldval, i8 %newval monotonic
  ret i8 %0

; CHECK:   AtomicCmpSwap8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   andi    $[[R8:[0-9]+]], $4, 255
; CHECK:   sllv    $[[R9:[0-9]+]], $[[R8]], $[[R4]]
; CHECK:   andi    $[[R10:[0-9]+]], $5, 255
; CHECK:   sllv    $[[R11:[0-9]+]], $[[R10]], $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R12:[0-9]+]], 0($[[R2]])
; CHECK:   and     $[[R13:[0-9]+]], $[[R12]], $[[R6]]
; CHECK:   bne     $[[R13]], $[[R9]], $[[BB1:[A-Z_0-9]+]]

; CHECK:   and     $[[R14:[0-9]+]], $[[R12]], $[[R7]]
; CHECK:   or      $[[R15:[0-9]+]], $[[R14]], $[[R11]]
; CHECK:   sc      $[[R15]], 0($[[R2]])
; CHECK:   beq     $[[R15]], $zero, $[[BB0]]

; CHECK:   $[[BB1]]:
; CHECK:   srlv    $[[R16:[0-9]+]], $[[R13]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

@countsint = common global i32 0, align 4

define i32 @CheckSync(i32 %v) nounwind noinline {
entry:
  %0 = atomicrmw add i32* @countsint, i32 %v seq_cst
  ret i32 %0 

; CHECK:   CheckSync:
; CHECK:   sync 0
; CHECK:   ll
; CHECK:   sc
; CHECK:   beq
; CHECK:   sync 0
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
  %0 = cmpxchg i32* @a, i32 1, i32 0 seq_cst
  %1 = icmp eq i32 %0, 1
  %conv = zext i1 %1 to i32
  ret i32 %conv
}
