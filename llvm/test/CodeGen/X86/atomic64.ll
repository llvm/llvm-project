; RUN: llc < %s -O0 -mtriple=x86_64-- -mcpu=corei7 -verify-machineinstrs | FileCheck %s --check-prefix X64

@sc64 = external global i64
@fsc64 = external global double

define void @atomic_fetch_add64() nounwind {
; X64-LABEL:   atomic_fetch_add64:
; X32-LABEL:   atomic_fetch_add64:
entry:
  %t1 = atomicrmw add  i64* @sc64, i64 1 acquire
; X64:       lock
; X64:       incq
  %t2 = atomicrmw add  i64* @sc64, i64 3 acquire
; X64:       lock
; X64:       addq $3
  %t3 = atomicrmw add  i64* @sc64, i64 5 acquire
; X64:       lock
; X64:       xaddq
  %t4 = atomicrmw add  i64* @sc64, i64 %t3 acquire
; X64:       lock
; X64:       addq
  ret void
; X64:       ret
}

define void @atomic_fetch_sub64() nounwind {
; X64-LABEL:   atomic_fetch_sub64:
; X32-LABEL:   atomic_fetch_sub64:
  %t1 = atomicrmw sub  i64* @sc64, i64 1 acquire
; X64:       lock
; X64:       decq
  %t2 = atomicrmw sub  i64* @sc64, i64 3 acquire
; X64:       lock
; X64:       subq $3
  %t3 = atomicrmw sub  i64* @sc64, i64 5 acquire
; X64:       lock
; X64:       xaddq
  %t4 = atomicrmw sub  i64* @sc64, i64 %t3 acquire
; X64:       lock
; X64:       subq
  ret void
; X64:       ret
}

define void @atomic_fetch_and64() nounwind {
; X64-LABEL:   atomic_fetch_and64:
; X32-LABEL:   atomic_fetch_and64:
  %t1 = atomicrmw and  i64* @sc64, i64 3 acquire
; X64:       lock
; X64:       andq $3
  %t2 = atomicrmw and  i64* @sc64, i64 5 acquire
; X64:       andl
; X64:       lock
; X64:       cmpxchgq
  %t3 = atomicrmw and  i64* @sc64, i64 %t2 acquire
; X64:       lock
; X64:       andq
  ret void
; X64:       ret
}

define void @atomic_fetch_or64() nounwind {
; X64-LABEL:   atomic_fetch_or64:
; X32-LABEL:   atomic_fetch_or64:
  %t1 = atomicrmw or   i64* @sc64, i64 3 acquire
; X64:       lock
; X64:       orq $3
  %t2 = atomicrmw or   i64* @sc64, i64 5 acquire
; X64:       orq
; X64:       lock
; X64:       cmpxchgq
  %t3 = atomicrmw or   i64* @sc64, i64 %t2 acquire
; X64:       lock
; X64:       orq
  ret void
; X64:       ret
}

define void @atomic_fetch_xor64() nounwind {
; X64-LABEL:   atomic_fetch_xor64:
; X32-LABEL:   atomic_fetch_xor64:
  %t1 = atomicrmw xor  i64* @sc64, i64 3 acquire
; X64:       lock
; X64:       xorq $3
  %t2 = atomicrmw xor  i64* @sc64, i64 5 acquire
; X64:       xorq
; X64:       lock
; X64:       cmpxchgq
  %t3 = atomicrmw xor  i64* @sc64, i64 %t2 acquire
; X64:       lock
; X64:       xorq
  ret void
; X64:       ret
}

define void @atomic_fetch_nand64(i64 %x) nounwind {
; X64-LABEL:   atomic_fetch_nand64:
; X32-LABEL:   atomic_fetch_nand64:
  %t1 = atomicrmw nand i64* @sc64, i64 %x acquire
; X64:       andq
; X64:       notq
; X64:       lock
; X64:       cmpxchgq
; X32:       andl
; X32:       andl
; X32:       notl
; X32:       notl
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_max64(i64 %x) nounwind {
; X64-LABEL:   atomic_fetch_max64:
; X32-LABEL:   atomic_fetch_max64:
  %t1 = atomicrmw max  i64* @sc64, i64 %x acquire
; X64:       subq
; X64:       cmov
; X64:       lock
; X64:       cmpxchgq

; X32:       cmpl
; X32:       cmpl
; X32:       cmov
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_min64(i64 %x) nounwind {
; X64-LABEL:   atomic_fetch_min64:
; X32-LABEL:   atomic_fetch_min64:
  %t1 = atomicrmw min  i64* @sc64, i64 %x acquire
; X64:       subq
; X64:       cmov
; X64:       lock
; X64:       cmpxchgq

; X32:       cmpl
; X32:       cmpl
; X32:       cmov
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umax64(i64 %x) nounwind {
; X64-LABEL:   atomic_fetch_umax64:
; X32-LABEL:   atomic_fetch_umax64:
  %t1 = atomicrmw umax i64* @sc64, i64 %x acquire
; X64:       subq
; X64:       cmov
; X64:       lock
; X64:       cmpxchgq

; X32:       cmpl
; X32:       cmpl
; X32:       cmov
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umin64(i64 %x) nounwind {
; X64-LABEL:   atomic_fetch_umin64:
; X32-LABEL:   atomic_fetch_umin64:
  %t1 = atomicrmw umin i64* @sc64, i64 %x acquire
; X64:       subq
; X64:       cmov
; X64:       lock
; X64:       cmpxchgq

; X32:       cmpl
; X32:       cmpl
; X32:       cmov
; X32:       cmov
; X32:       cmov
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_cmpxchg64() nounwind {
; X64-LABEL:   atomic_fetch_cmpxchg64:
; X32-LABEL:   atomic_fetch_cmpxchg64:
  %t1 = cmpxchg i64* @sc64, i64 0, i64 1 acquire acquire
; X64:       lock
; X64:       cmpxchgq
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_store64(i64 %x) nounwind {
; X64-LABEL:   atomic_fetch_store64:
; X32-LABEL:   atomic_fetch_store64:
  store atomic i64 %x, i64* @sc64 release, align 8
; X64-NOT:   lock
; X64:       movq
; X32:       lock
; X32:       cmpxchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_swap64(i64 %x) nounwind {
; X64-LABEL:   atomic_fetch_swap64:
; X32-LABEL:   atomic_fetch_swap64:
  %t1 = atomicrmw xchg i64* @sc64, i64 %x acquire
; X64-NOT:   lock
; X64:       xchgq
; X32:       lock
; X32:       xchg8b
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_swapf64(double %x) nounwind {
; X64-LABEL:   atomic_fetch_swapf64:
; X32-LABEL:   atomic_fetch_swapf64:
  %t1 = atomicrmw xchg double* @fsc64, double %x acquire
; X64-NOT:   lock
; X64:       xchgq
; X32:       lock
; X32:       xchg8b
  ret void
; X64:       ret
; X32:       ret
}
