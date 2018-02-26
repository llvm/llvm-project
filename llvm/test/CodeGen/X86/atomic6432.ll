; RUN: llc < %s -O0 -mtriple=i686-- -mcpu=corei7 -verify-machineinstrs | FileCheck %s --check-prefix X32

@sc64 = external global i64

define void @atomic_fetch_add64() nounwind {
; X32-LABEL:   atomic_fetch_add64:
entry:
  %t1 = atomicrmw add  i64* @sc64, i64 1 acquire
; X32:       addl
; X32:       adcl
; X32:       lock
; X32:       cmpxchg8b
  %t2 = atomicrmw add  i64* @sc64, i64 3 acquire
; X32:       addl
; X32:       adcl
; X32:       lock
; X32:       cmpxchg8b
  %t3 = atomicrmw add  i64* @sc64, i64 5 acquire
; X32:       addl
; X32:       adcl
; X32:       lock
; X32:       cmpxchg8b
  %t4 = atomicrmw add  i64* @sc64, i64 %t3 acquire
; X32:       addl
; X32:       adcl
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_sub64() nounwind {
; X32-LABEL:   atomic_fetch_sub64:
  %t1 = atomicrmw sub  i64* @sc64, i64 1 acquire
; X32:       addl $-1
; X32:       adcl $-1
; X32:       lock
; X32:       cmpxchg8b
  %t2 = atomicrmw sub  i64* @sc64, i64 3 acquire
; X32:       addl $-3
; X32:       adcl $-1
; X32:       lock
; X32:       cmpxchg8b
  %t3 = atomicrmw sub  i64* @sc64, i64 5 acquire
; X32:       addl $-5
; X32:       adcl $-1
; X32:       lock
; X32:       cmpxchg8b
  %t4 = atomicrmw sub  i64* @sc64, i64 %t3 acquire
; X32:       subl
; X32:       sbbl
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_and64() nounwind {
; X32-LABEL:   atomic_fetch_and64:
  %t1 = atomicrmw and  i64* @sc64, i64 3 acquire
; X32:       andl $3
; X32-NOT:       andl
; X32:       lock
; X32:       cmpxchg8b
  %t2 = atomicrmw and  i64* @sc64, i64 4294967297 acquire
; X32:       andl $1
; X32:       andl $1
; X32:       lock
; X32:       cmpxchg8b
  %t3 = atomicrmw and  i64* @sc64, i64 %t2 acquire
; X32:       andl
; X32:       andl
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_or64() nounwind {
; X32-LABEL:   atomic_fetch_or64:
  %t1 = atomicrmw or   i64* @sc64, i64 3 acquire
; X32:       orl $3
; X32-NOT:       orl
; X32:       lock
; X32:       cmpxchg8b
  %t2 = atomicrmw or   i64* @sc64, i64 4294967297 acquire
; X32:       orl $1
; X32:       orl $1
; X32:       lock
; X32:       cmpxchg8b
  %t3 = atomicrmw or   i64* @sc64, i64 %t2 acquire
; X32:       orl
; X32:       orl
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_xor64() nounwind {
; X32-LABEL:   atomic_fetch_xor64:
  %t1 = atomicrmw xor  i64* @sc64, i64 3 acquire
; X32:       xorl
; X32-NOT:       xorl
; X32:       lock
; X32:       cmpxchg8b
  %t2 = atomicrmw xor  i64* @sc64, i64 4294967297 acquire
; X32:       xorl $1
; X32:       xorl $1
; X32:       lock
; X32:       cmpxchg8b
  %t3 = atomicrmw xor  i64* @sc64, i64 %t2 acquire
; X32:       xorl
; X32:       xorl
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_nand64(i64 %x) nounwind {
; X32-LABEL:   atomic_fetch_nand64:
  %t1 = atomicrmw nand i64* @sc64, i64 %x acquire
; X32:       andl
; X32:       andl
; X32:       notl
; X32:       notl
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_max64(i64 %x) nounwind {
; X32-LABEL:   atomic_fetch_max64:
  %t1 = atomicrmw max  i64* @sc64, i64 %x acquire
; X32:       subl
; X32:       subl
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_min64(i64 %x) nounwind {
; X32-LABEL:   atomic_fetch_min64:
  %t1 = atomicrmw min  i64* @sc64, i64 %x acquire
; X32:       subl
; X32:       subl
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_umax64(i64 %x) nounwind {
; X32-LABEL:   atomic_fetch_umax64:
  %t1 = atomicrmw umax i64* @sc64, i64 %x acquire
; X32:       subl
; X32:       subl
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_umin64(i64 %x) nounwind {
; X32-LABEL:   atomic_fetch_umin64:
  %t1 = atomicrmw umin i64* @sc64, i64 %x acquire
; X32:       subl
; X32:       subl
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_cmpxchg64() nounwind {
; X32-LABEL:   atomic_fetch_cmpxchg64:
  %t1 = cmpxchg i64* @sc64, i64 0, i64 1 acquire acquire
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_store64(i64 %x) nounwind {
; X32-LABEL:   atomic_fetch_store64:
  store atomic i64 %x, i64* @sc64 release, align 8
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X32:       ret
}

define void @atomic_fetch_swap64(i64 %x) nounwind {
; X32-LABEL:   atomic_fetch_swap64:
  %t1 = atomicrmw xchg i64* @sc64, i64 %x acquire
; X32:       lock
; X32:       xchg8b
  ret void
; X32:       ret
}
