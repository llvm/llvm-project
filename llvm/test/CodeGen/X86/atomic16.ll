; RUN: llc < %s -O0 -mtriple=x86_64-unknown-unknown -mcpu=corei7 -verify-machineinstrs -show-mc-encoding | FileCheck %s --check-prefix X64
; RUN: llc < %s -O0 -mtriple=i386-unknown-unknown -mcpu=corei7 -verify-machineinstrs | FileCheck %s --check-prefix X32

@sc16 = external global i16

define void @atomic_fetch_add16() nounwind {
; X64-LABEL:   atomic_fetch_add16
; X32-LABEL:   atomic_fetch_add16
entry:
; 32-bit
  %t1 = atomicrmw add  i16* @sc16, i16 1 acquire
; X64:       lock
; X64:       incw
; X32:       lock
; X32:       incw
  %t2 = atomicrmw add  i16* @sc16, i16 3 acquire
; X64:       lock
; X64:       addw $3, {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       addw $3
  %t3 = atomicrmw add  i16* @sc16, i16 5 acquire
; X64:       lock
; X64:       xaddw {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       xaddw
  %t4 = atomicrmw add  i16* @sc16, i16 %t3 acquire
; X64:       lock
; X64:       addw {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       addw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_sub16() nounwind {
; X64-LABEL:   atomic_fetch_sub16
; X32-LABEL:   atomic_fetch_sub16
  %t1 = atomicrmw sub  i16* @sc16, i16 1 acquire
; X64:       lock
; X64:       decw
; X32:       lock
; X32:       decw
  %t2 = atomicrmw sub  i16* @sc16, i16 3 acquire
; X64:       lock
; X64:       subw $3, {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       subw $3
  %t3 = atomicrmw sub  i16* @sc16, i16 5 acquire
; X64:       lock
; X64:       xaddw {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       xaddw
  %t4 = atomicrmw sub  i16* @sc16, i16 %t3 acquire
; X64:       lock
; X64:       subw {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       subw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_and16() nounwind {
; X64-LABEL:   atomic_fetch_and16
; X32-LABEL:   atomic_fetch_and16
  %t1 = atomicrmw and  i16* @sc16, i16 3 acquire
; X64:       lock
; X64:       andw $3, {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       andw $3
  %t2 = atomicrmw and  i16* @sc16, i16 5 acquire
; X64:       andl
; X64:       lock
; X64:       cmpxchgw
; X32:       andl
; X32:       lock
; X32:       cmpxchgw
  %t3 = atomicrmw and  i16* @sc16, i16 %t2 acquire
; X64:       lock
; X64:       andw {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       andw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_or16() nounwind {
; X64-LABEL:   atomic_fetch_or16
; X32-LABEL:   atomic_fetch_or16
  %t1 = atomicrmw or   i16* @sc16, i16 3 acquire
; X64:       lock
; X64:       orw $3, {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       orw $3
  %t2 = atomicrmw or   i16* @sc16, i16 5 acquire
; X64:       orl
; X64:       lock
; X64:       cmpxchgw
; X32:       orl
; X32:       lock
; X32:       cmpxchgw
  %t3 = atomicrmw or   i16* @sc16, i16 %t2 acquire
; X64:       lock
; X64:       orw {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       orw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_xor16() nounwind {
; X64-LABEL:   atomic_fetch_xor16
; X32-LABEL:   atomic_fetch_xor16
  %t1 = atomicrmw xor  i16* @sc16, i16 3 acquire
; X64:       lock
; X64:       xorw $3, {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       xorw $3
  %t2 = atomicrmw xor  i16* @sc16, i16 5 acquire
; X64:       xorl
; X64:       lock
; X64:       cmpxchgw
; X32:       xorl
; X32:       lock
; X32:       cmpxchgw
  %t3 = atomicrmw xor  i16* @sc16, i16 %t2 acquire
; X64:       lock
; X64:       xorw {{.*}} # encoding: [0x66,0xf0
; X32:       lock
; X32:       xorw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_nand16(i16 %x) nounwind {
; X64-LABEL:   atomic_fetch_nand16
; X32-LABEL:   atomic_fetch_nand16
  %t1 = atomicrmw nand i16* @sc16, i16 %x acquire
; X64:       andl
; X64:       notl
; X64:       lock
; X64:       cmpxchgw
; X32:       andl
; X32:       notl
; X32:       lock
; X32:       cmpxchgw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_max16(i16 %x) nounwind {
; X64-LABEL:   atomic_fetch_max16
; X32-LABEL:   atomic_fetch_max16
  %t1 = atomicrmw max  i16* @sc16, i16 %x acquire
; X64:       movw
; X64:       movw
; X64:       subw
; X64:       cmov
; X64:       lock
; X64:       cmpxchgw

; X32:       movw
; X32:       movw
; X32:       subw
; X32:       cmov
; X32:       lock
; X32:       cmpxchgw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_min16(i16 %x) nounwind {
; X64-LABEL:   atomic_fetch_min16
; X32-LABEL:   atomic_fetch_min16
  %t1 = atomicrmw min  i16* @sc16, i16 %x acquire
; X64:       movw
; X64:       movw
; X64:       subw
; X64:       cmov
; X64:       lock
; X64:       cmpxchgw

; X32:       movw
; X32:       movw
; X32:       subw
; X32:       cmov
; X32:       lock
; X32:       cmpxchgw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umax16(i16 %x) nounwind {
; X64-LABEL:   atomic_fetch_umax16
; X32-LABEL:   atomic_fetch_umax16
  %t1 = atomicrmw umax i16* @sc16, i16 %x acquire
; X64:       movw
; X64:       movw
; X64:       subw
; X64:       cmov
; X64:       lock
; X64:       cmpxchgw

; X32:       movw
; X32:       movw
; X32:       subw
; X32:       cmov
; X32:       lock
; X32:       cmpxchgw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_umin16(i16 %x) nounwind {
; X64-LABEL:   atomic_fetch_umin16
; X32-LABEL:   atomic_fetch_umin16
  %t1 = atomicrmw umin i16* @sc16, i16 %x acquire
; X64:       movw
; X64:       movw
; X64:       subw
; X64:       cmov
; X64:       lock
; X64:       cmpxchgw

; X32:       movw
; X32:       movw
; X32:       subw
; X32:       cmov
; X32:       lock
; X32:       cmpxchgw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_cmpxchg16() nounwind {
  %t1 = cmpxchg i16* @sc16, i16 0, i16 1 acquire acquire
; X64:       lock
; X64:       cmpxchgw
; X32:       lock
; X32:       cmpxchgw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_store16(i16 %x) nounwind {
  store atomic i16 %x, i16* @sc16 release, align 4
; X64-NOT:   lock
; X64:       movw
; X32-NOT:   lock
; X32:       movw
  ret void
; X64:       ret
; X32:       ret
}

define void @atomic_fetch_swap16(i16 %x) nounwind {
  %t1 = atomicrmw xchg i16* @sc16, i16 %x acquire
; X64-NOT:   lock
; X64:       xchgw
; X32-NOT:   lock
; X32:       xchgw
  ret void
; X64:       ret
; X32:       ret
}
