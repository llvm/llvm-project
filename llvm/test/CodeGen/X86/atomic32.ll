; RUN: llc < %s -O0 -march=x86-64 -mcpu=corei7 | FileCheck %s --check-prefix X64
; RUN: llc < %s -O0 -march=x86 -mcpu=corei7 | FileCheck %s --check-prefix X32

@sc32 = external global i32

define void @atomic_fetch_add32() nounwind {
; X64:   atomic_fetch_add32
; X32:   atomic_fetch_add32
entry:
; 32-bit
  %t1 = atomicrmw add  i32* @sc32, i32 1 acquire
; X64:       lock
; X64:       incl
; X32:       lock
; X32:       incl
  %t2 = atomicrmw add  i32* @sc32, i32 3 acquire
; X64:       lock
; X64:       addl $3
; X32:       lock
; X32:       addl $3
  %t3 = atomicrmw add  i32* @sc32, i32 5 acquire
; X64:       lock
; X64:       xaddl
; X32:       lock
; X32:       xaddl
  %t4 = atomicrmw add  i32* @sc32, i32 %t3 acquire
; X64:       lock
; X64:       addl
; X32:       lock
; X32:       addl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_sub32() nounwind {
; X64:   atomic_fetch_sub32
; X32:   atomic_fetch_sub32
  %t1 = atomicrmw sub  i32* @sc32, i32 1 acquire
; X64:       lock
; X64:       decl
; X32:       lock
; X32:       decl
  %t2 = atomicrmw sub  i32* @sc32, i32 3 acquire
; X64:       lock
; X64:       subl $3
; X32:       lock
; X32:       subl $3
  %t3 = atomicrmw sub  i32* @sc32, i32 5 acquire
; X64:       lock
; X64:       xaddl
; X32:       lock
; X32:       xaddl
  %t4 = atomicrmw sub  i32* @sc32, i32 %t3 acquire
; X64:       lock
; X64:       subl
; X32:       lock
; X32:       subl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_and32() nounwind {
; X64:   atomic_fetch_and32
; X32:   atomic_fetch_and32
  %t1 = atomicrmw and  i32* @sc32, i32 3 acquire
; X64:       lock
; X64:       andl $3
; X32:       lock
; X32:       andl $3
  %t2 = atomicrmw and  i32* @sc32, i32 5 acquire
; X64:       andl
; X64:       lock
; X64:       cmpxchgl
; X32:       andl
; X32:       lock
; X32:       cmpxchgl
  %t3 = atomicrmw and  i32* @sc32, i32 %t2 acquire
; X64:       lock
; X64:       andl
; X32:       lock
; X32:       andl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_or32() nounwind {
; X64:   atomic_fetch_or32
; X32:   atomic_fetch_or32
  %t1 = atomicrmw or   i32* @sc32, i32 3 acquire
; X64:       lock
; X64:       orl $3
; X32:       lock
; X32:       orl $3
  %t2 = atomicrmw or   i32* @sc32, i32 5 acquire
; X64:       orl
; X64:       lock
; X64:       cmpxchgl
; X32:       orl
; X32:       lock
; X32:       cmpxchgl
  %t3 = atomicrmw or   i32* @sc32, i32 %t2 acquire
; X64:       lock
; X64:       orl
; X32:       lock
; X32:       orl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_xor32() nounwind {
; X64:   atomic_fetch_xor32
; X32:   atomic_fetch_xor32
  %t1 = atomicrmw xor  i32* @sc32, i32 3 acquire
; X64:       lock
; X64:       xorl $3
; X32:       lock
; X32:       xorl $3
  %t2 = atomicrmw xor  i32* @sc32, i32 5 acquire
; X64:       xorl
; X64:       lock
; X64:       cmpxchgl
; X32:       xorl
; X32:       lock
; X32:       cmpxchgl
  %t3 = atomicrmw xor  i32* @sc32, i32 %t2 acquire
; X64:       lock
; X64:       xorl
; X32:       lock
; X32:       xorl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_nand32(i32 %x) nounwind {
; X64:   atomic_fetch_nand32
; X32:   atomic_fetch_nand32
  %t1 = atomicrmw nand i32* @sc32, i32 %x acquire
; X64:       andl
; X64:       notl
; X64:       lock
; X64:       cmpxchgl
; X32:       andl
; X32:       notl
; X32:       lock
; X32:       cmpxchgl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_max32(i32 %x) nounwind {
  %t1 = atomicrmw max  i32* @sc32, i32 %x acquire
; X64:       cmpl
; X64:       cmov
; X64:       lock
; X64:       cmpxchgl

; X32:       cmpl
; X32:       cmov
; X32:       lock
; X32:       cmpxchgl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_min32(i32 %x) nounwind {
  %t1 = atomicrmw min  i32* @sc32, i32 %x acquire
; X64:       cmpl
; X64:       cmov
; X64:       lock
; X64:       cmpxchgl

; X32:       cmpl
; X32:       cmov
; X32:       lock
; X32:       cmpxchgl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umax32(i32 %x) nounwind {
  %t1 = atomicrmw umax i32* @sc32, i32 %x acquire
; X64:       cmpl
; X64:       cmov
; X64:       lock
; X64:       cmpxchgl

; X32:       cmpl
; X32:       cmov
; X32:       lock
; X32:       cmpxchgl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umin32(i32 %x) nounwind {
  %t1 = atomicrmw umin i32* @sc32, i32 %x acquire
; X64:       cmpl
; X64:       cmov
; X64:       lock
; X64:       cmpxchgl
; X32:       cmpl
; X32:       cmov
; X32:       lock
; X32:       cmpxchgl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_cmpxchg32() nounwind {
  %t1 = cmpxchg i32* @sc32, i32 0, i32 1 acquire
; X64:       lock
; X64:       cmpxchgl
; X32:       lock
; X32:       cmpxchgl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_store32(i32 %x) nounwind {
  store atomic i32 %x, i32* @sc32 release, align 4
; X64-NOT:   lock
; X64:       movl
; X32-NOT:   lock
; X32:       movl
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_swap32(i32 %x) nounwind {
  %t1 = atomicrmw xchg i32* @sc32, i32 %x acquire
; X64-NOT:   lock
; X64:       xchgl
; X32-NOT:   lock
; X32:       xchgl
  ret void
; X64:       ret
; X32:       ret
}
