; RUN: llc < %s -O0 -mtriple=x86_64-- -mcpu=corei7 -verify-machineinstrs | FileCheck %s --check-prefix X64
; RUN: llc < %s -O0 -mtriple=i686-- -mcpu=corei7 -verify-machineinstrs | FileCheck %s --check-prefix X32

@sc8 = external dso_local global i8

define void @atomic_fetch_add8() nounwind {
; X64-LABEL:   atomic_fetch_add8:
; X32-LABEL:   atomic_fetch_add8:
entry:
; 32-bit
  %t1 = atomicrmw add  i8* @sc8, i8 1 acquire
; X64:       lock
; X64:       incb
; X32:       lock
; X32:       incb
  %t2 = atomicrmw add  i8* @sc8, i8 3 acquire
; X64:       lock
; X64:       addb $3
; X32:       lock
; X32:       addb $3
  %t3 = atomicrmw add  i8* @sc8, i8 5 acquire
; X64:       lock
; X64:       xaddb
; X32:       lock
; X32:       xaddb
  %t4 = atomicrmw add  i8* @sc8, i8 %t3 acquire
; X64:       lock
; X64:       addb
; X32:       lock
; X32:       addb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_sub8() nounwind {
; X64-LABEL:   atomic_fetch_sub8:
; X32-LABEL:   atomic_fetch_sub8:
  %t1 = atomicrmw sub  i8* @sc8, i8 1 acquire
; X64:       lock
; X64:       decb
; X32:       lock
; X32:       decb
  %t2 = atomicrmw sub  i8* @sc8, i8 3 acquire
; X64:       lock
; X64:       subb $3
; X32:       lock
; X32:       subb $3
  %t3 = atomicrmw sub  i8* @sc8, i8 5 acquire
; X64:       lock
; X64:       xaddb
; X32:       lock
; X32:       xaddb
  %t4 = atomicrmw sub  i8* @sc8, i8 %t3 acquire
; X64:       lock
; X64:       subb
; X32:       lock
; X32:       subb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_and8() nounwind {
; X64-LABEL:   atomic_fetch_and8:
; X32-LABEL:   atomic_fetch_and8:
  %t1 = atomicrmw and  i8* @sc8, i8 3 acquire
; X64:       lock
; X64:       andb $3
; X32:       lock
; X32:       andb $3
  %t2 = atomicrmw and  i8* @sc8, i8 5 acquire
; X64:       andb
; X64:       lock
; X64:       cmpxchgb
; X32:       andb
; X32:       lock
; X32:       cmpxchgb
  %t3 = atomicrmw and  i8* @sc8, i8 %t2 acquire
; X64:       lock
; X64:       andb
; X32:       lock
; X32:       andb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_or8() nounwind {
; X64-LABEL:   atomic_fetch_or8:
; X32-LABEL:   atomic_fetch_or8:
  %t1 = atomicrmw or   i8* @sc8, i8 3 acquire
; X64:       lock
; X64:       orb $3
; X32:       lock
; X32:       orb $3
  %t2 = atomicrmw or   i8* @sc8, i8 5 acquire
; X64:       orb
; X64:       lock
; X64:       cmpxchgb
; X32:       orb
; X32:       lock
; X32:       cmpxchgb
  %t3 = atomicrmw or   i8* @sc8, i8 %t2 acquire
; X64:       lock
; X64:       orb
; X32:       lock
; X32:       orb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_xor8() nounwind {
; X64-LABEL:   atomic_fetch_xor8:
; X32-LABEL:   atomic_fetch_xor8:
  %t1 = atomicrmw xor  i8* @sc8, i8 3 acquire
; X64:       lock
; X64:       xorb $3
; X32:       lock
; X32:       xorb $3
  %t2 = atomicrmw xor  i8* @sc8, i8 5 acquire
; X64:       xorb
; X64:       lock
; X64:       cmpxchgb
; X32:       xorb
; X32:       lock
; X32:       cmpxchgb
  %t3 = atomicrmw xor  i8* @sc8, i8 %t2 acquire
; X64:       lock
; X64:       xorb
; X32:       lock
; X32:       xorb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_nand8(i8 %x) nounwind {
; X64-LABEL:   atomic_fetch_nand8:
; X32-LABEL:   atomic_fetch_nand8:
  %t1 = atomicrmw nand i8* @sc8, i8 %x acquire
; X64:       andb
; X64:       notb
; X64:       lock
; X64:       cmpxchgb
; X32:       andb
; X32:       notb
; X32:       lock
; X32:       cmpxchgb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_max8(i8 %x) nounwind {
; X64-LABEL:   atomic_fetch_max8:
; X32-LABEL:   atomic_fetch_max8:
  %t1 = atomicrmw max  i8* @sc8, i8 %x acquire
; X64:       movb
; X64:       movb
; X64:       subb
; X64:       lock
; X64:       cmpxchgb

; X32:       movb
; X32:       movb
; X32:       subb
; X32:       lock
; X32:       cmpxchgb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_min8(i8 %x) nounwind {
; X64-LABEL:   atomic_fetch_min8:
; X32-LABEL:   atomic_fetch_min8:
  %t1 = atomicrmw min  i8* @sc8, i8 %x acquire
; X64:       movb
; X64:       movb
; X64:       subb
; X64:       lock
; X64:       cmpxchgb

; X32:       movb
; X32:       movb
; X32:       subb
; X32:       lock
; X32:       cmpxchgb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umax8(i8 %x) nounwind {
; X64-LABEL:   atomic_fetch_umax8:
; X32-LABEL:   atomic_fetch_umax8:
  %t1 = atomicrmw umax i8* @sc8, i8 %x acquire
; X64:       movb
; X64:       movb
; X64:       subb
; X64:       lock
; X64:       cmpxchgb

; X32:       movb
; X32:       movb
; X32:       subb
; X32:       lock
; X32:       cmpxchgb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umin8(i8 %x) nounwind {
; X64-LABEL:   atomic_fetch_umin8:
; X32-LABEL:   atomic_fetch_umin8:
  %t1 = atomicrmw umin i8* @sc8, i8 %x acquire
; X64:       movb
; X64:       movb
; X64:       subb
; X64:       lock
; X64:       cmpxchgb

; X32:       movb
; X32:       movb
; X32:       subb
; X32:       lock
; X32:       cmpxchgb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_cmpxchg8() nounwind {
; X64-LABEL:   atomic_fetch_cmpxchg8:
; X32-LABEL:   atomic_fetch_cmpxchg8:
  %t1 = cmpxchg i8* @sc8, i8 0, i8 1 acquire acquire
; X64:       lock
; X64:       cmpxchgb
; X32:       lock
; X32:       cmpxchgb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_store8(i8 %x) nounwind {
; X64-LABEL:   atomic_fetch_store8:
; X32-LABEL:   atomic_fetch_store8:
  store atomic i8 %x, i8* @sc8 release, align 4
; X64-NOT:   lock
; X64:       movb
; X32-NOT:   lock
; X32:       movb
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_swap8(i8 %x) nounwind {
; X64-LABEL:   atomic_fetch_swap8:
; X32-LABEL:   atomic_fetch_swap8:
  %t1 = atomicrmw xchg i8* @sc8, i8 %x acquire
; X64-NOT:   lock
; X64:       xchgb
; X32-NOT:   lock
; X32:       xchgb
  ret void
; X64:       ret
; X32:       ret
}
