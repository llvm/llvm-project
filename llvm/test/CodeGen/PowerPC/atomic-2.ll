; RUN: llc -verify-machineinstrs < %s -ppc-asm-full-reg-names -mtriple=ppc64-- | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE
; RUN: llc -verify-machineinstrs < %s -ppc-asm-full-reg-names -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc -verify-machineinstrs < %s -ppc-asm-full-reg-names -mtriple=ppc64-- -mcpu=pwr7 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -ppc-asm-full-reg-names -mtriple=ppc64-- -mcpu=pwr8 | FileCheck %s -check-prefix=CHECK-P8U

define i64 @exchange_and_add(ptr %mem, i64 %val) nounwind {
; CHECK-LABEL: exchange_and_add:
; CHECK: ldarx
  %tmp = atomicrmw add ptr %mem, i64 %val monotonic
; CHECK: stdcx.
  ret i64 %tmp
}

define i8 @exchange_and_add8(ptr %mem, i8 %val) nounwind {
; CHECK-LABEL: exchange_and_add8:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lbarx
  %tmp = atomicrmw add ptr %mem, i8 %val monotonic
; CHECK-P8U: stbcx.
  ret i8 %tmp
}

define i16 @exchange_and_add16(ptr %mem, i16 %val) nounwind {
; CHECK-LABEL: exchange_and_add16:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lharx
  %tmp = atomicrmw add ptr %mem, i16 %val monotonic
; CHECK-P8U: sthcx.
  ret i16 %tmp
}

define i64 @exchange_and_cmp(ptr %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp:
; CHECK: ldarx
  %tmppair = cmpxchg ptr %mem, i64 0, i64 1 monotonic monotonic
  %tmp = extractvalue { i64, i1 } %tmppair, 0
; CHECK: stdcx.
  ret i64 %tmp
}

define i8 @exchange_and_cmp8(ptr %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp8:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lbarx
  %tmppair = cmpxchg ptr %mem, i8 0, i8 1 monotonic monotonic
  %tmp = extractvalue { i8, i1 } %tmppair, 0
; CHECK-P8U: stbcx.
  ret i8 %tmp
}

define i16 @exchange_and_cmp16(ptr %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp16:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lharx
  %tmppair = cmpxchg ptr %mem, i16 0, i16 1 monotonic monotonic
  %tmp = extractvalue { i16, i1 } %tmppair, 0
; CHECK-P8U: sthcx.
  ret i16 %tmp
}

define i64 @exchange(ptr %mem, i64 %val) nounwind {
; CHECK-LABEL: exchange:
; CHECK: ldarx
  %tmp = atomicrmw xchg ptr %mem, i64 1 monotonic
; CHECK: stdcx.
  ret i64 %tmp
}

define i8 @exchange8(ptr %mem, i8 %val) nounwind {
; CHECK-LABEL: exchange8:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lbarx
  %tmp = atomicrmw xchg ptr %mem, i8 1 monotonic
; CHECK-P8U: stbcx.
  ret i8 %tmp
}

define i16 @exchange16(ptr %mem, i16 %val) nounwind {
; CHECK-LABEL: exchange16:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lharx
  %tmp = atomicrmw xchg ptr %mem, i16 1 monotonic
; CHECK-P8U: sthcx.
  ret i16 %tmp
}

define void @atomic_store(ptr %mem, i64 %val) nounwind {
entry:
; CHECK: @atomic_store
  store atomic i64 %val, ptr %mem release, align 64
; CHECK: lwsync
; CHECK-NOT: stdcx
; CHECK: std
  ret void
}

define i64 @atomic_load(ptr %mem) nounwind {
entry:
; CHECK: @atomic_load
  %tmp = load atomic i64, ptr %mem acquire, align 64
; CHECK-NOT: ldarx
; CHECK: ld [[VAL:r[0-9]+]]
; CHECK: cmpd [[CR:cr[0-9]+]], [[VAL]], [[VAL]]
; CHECK: bne- [[CR]], .+4
; CHECK: isync
  ret i64 %tmp
}

