; RUN: llc < %s -O0 -mtriple=x86_64-- -mcpu=corei7 -verify-machineinstrs | FileCheck %s -check-prefix=WITH-CMOV
; RUN: llc < %s -O0 -mtriple=i686-- -mcpu=corei7 -verify-machineinstrs | FileCheck %s -check-prefix=WITH-CMOV
; RUN: llc < %s -O0 -mtriple=i686-- -mcpu=corei7 -mattr=-cmov -verify-machineinstrs | FileCheck %s --check-prefix NOCMOV

@sc32 = external global i32

define void @atomic_fetch_add32() nounwind {
; WITH-CMOV-LABEL:   atomic_fetch_add32:
entry:
; 32-bit
  %t1 = atomicrmw add  i32* @sc32, i32 1 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       incl
  %t2 = atomicrmw add  i32* @sc32, i32 3 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       addl $3
  %t3 = atomicrmw add  i32* @sc32, i32 5 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       xaddl
  %t4 = atomicrmw add  i32* @sc32, i32 %t3 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       addl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_sub32() nounwind {
; WITH-CMOV-LABEL:   atomic_fetch_sub32:
  %t1 = atomicrmw sub  i32* @sc32, i32 1 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       decl
  %t2 = atomicrmw sub  i32* @sc32, i32 3 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       subl $3
  %t3 = atomicrmw sub  i32* @sc32, i32 5 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       xaddl
  %t4 = atomicrmw sub  i32* @sc32, i32 %t3 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       subl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_and32() nounwind {
; WITH-CMOV-LABEL:   atomic_fetch_and32:
  %t1 = atomicrmw and  i32* @sc32, i32 3 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       andl $3
  %t2 = atomicrmw and  i32* @sc32, i32 5 acquire
; WITH-CMOV:       andl
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl
  %t3 = atomicrmw and  i32* @sc32, i32 %t2 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       andl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_or32() nounwind {
; WITH-CMOV-LABEL:   atomic_fetch_or32:
  %t1 = atomicrmw or   i32* @sc32, i32 3 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       orl $3
  %t2 = atomicrmw or   i32* @sc32, i32 5 acquire
; WITH-CMOV:       orl
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl
  %t3 = atomicrmw or   i32* @sc32, i32 %t2 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       orl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_xor32() nounwind {
; WITH-CMOV-LABEL:   atomic_fetch_xor32:
  %t1 = atomicrmw xor  i32* @sc32, i32 3 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       xorl $3
  %t2 = atomicrmw xor  i32* @sc32, i32 5 acquire
; WITH-CMOV:       xorl
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl
  %t3 = atomicrmw xor  i32* @sc32, i32 %t2 acquire
; WITH-CMOV:       lock
; WITH-CMOV:       xorl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_nand32(i32 %x) nounwind {
; WITH-CMOV-LABEL:   atomic_fetch_nand32:
  %t1 = atomicrmw nand i32* @sc32, i32 %x acquire
; WITH-CMOV:       andl
; WITH-CMOV:       notl
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_max32(i32 %x) nounwind {
; WITH-CMOV-LABEL: atomic_fetch_max32:

  %t1 = atomicrmw max  i32* @sc32, i32 %x acquire
; WITH-CMOV:       subl
; WITH-CMOV:       cmov
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl

; NOCMOV:    subl
; NOCMOV:    jge
; NOCMOV:    lock
; NOCMOV:    cmpxchgl
  ret void
; WITH-CMOV:       ret
; NOCMOV:    ret
}

define void @atomic_fetch_min32(i32 %x) nounwind {
; WITH-CMOV-LABEL: atomic_fetch_min32:
; NOCMOV-LABEL: atomic_fetch_min32:

  %t1 = atomicrmw min  i32* @sc32, i32 %x acquire
; WITH-CMOV:       subl
; WITH-CMOV:       cmov
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl

; NOCMOV:    subl
; NOCMOV:    jle
; NOCMOV:    lock
; NOCMOV:    cmpxchgl
  ret void
; WITH-CMOV:       ret
; NOCMOV:    ret
}

define void @atomic_fetch_umax32(i32 %x) nounwind {
; WITH-CMOV-LABEL: atomic_fetch_umax32:
; NOCMOV-LABEL: atomic_fetch_umax32:

  %t1 = atomicrmw umax i32* @sc32, i32 %x acquire
; WITH-CMOV:       subl
; WITH-CMOV:       cmov
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl

; NOCMOV:    subl
; NOCMOV:    ja
; NOCMOV:    lock
; NOCMOV:    cmpxchgl
  ret void
; WITH-CMOV:       ret
; NOCMOV:    ret
}

define void @atomic_fetch_umin32(i32 %x) nounwind {
; WITH-CMOV-LABEL: atomic_fetch_umin32:
; NOCMOV-LABEL: atomic_fetch_umin32:

  %t1 = atomicrmw umin i32* @sc32, i32 %x acquire
; WITH-CMOV:       subl
; WITH-CMOV:       cmov
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl

; NOCMOV:    subl
; NOCMOV:    jb
; NOCMOV:    lock
; NOCMOV:    cmpxchgl
  ret void
; WITH-CMOV:       ret
; NOCMOV:    ret
}

define void @atomic_fetch_cmpxchg32() nounwind {
; WITH-CMOV-LABEL: atomic_fetch_cmpxchg32:

  %t1 = cmpxchg i32* @sc32, i32 0, i32 1 acquire acquire
; WITH-CMOV:       lock
; WITH-CMOV:       cmpxchgl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_store32(i32 %x) nounwind {
; WITH-CMOV-LABEL: atomic_fetch_store32:

  store atomic i32 %x, i32* @sc32 release, align 4
; WITH-CMOV-NOT:   lock
; WITH-CMOV:       movl
  ret void
; WITH-CMOV:       ret
}

define void @atomic_fetch_swap32(i32 %x) nounwind {
; WITH-CMOV-LABEL: atomic_fetch_swap32:

  %t1 = atomicrmw xchg i32* @sc32, i32 %x acquire
; WITH-CMOV-NOT:   lock
; WITH-CMOV:       xchgl
  ret void
; WITH-CMOV:       ret
}
